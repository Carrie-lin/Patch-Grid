from datetime import datetime
import os

import logging as log
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from loss.patch_loss import compute_patch_loss
from loss.merge_loss import compute_merge_loss
import torch.optim.lr_scheduler
from dataset.dataset import SurfaceDataset
from nets.OctreeSDF import OctreeSDF
from utils import PerfTimer, read_json
from loss.merge_utils import map_patch_grid_dict


print_time = False

class Trainer(object):

    def __init__(self, args, args_str):

        self.args = args 
        self.args_str = args_str
        
        self.args.epochs += 1


        # Set device to use
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        device_name = torch.cuda.get_device_name(device = self.device)
        log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        self.log_dict = {}

        # Initialize
        self.set_adapt_grids()
        self.set_update_patch_list()
        self.set_dataset()
        self.set_network()
        self.set_optimizer()
        self.set_lr_scheduler()
        if self.args.fix_decoder:
            self.net.freeze_decoder() 
        self.set_logger()


    ## get the feature volume grids number for each patch
    def set_adapt_grids(self):

        self.load_adaptive_grids()

        ## for the case of updating a shape
        if self.args.update:
            self.load_adaptive_grids(self.args.follow_shape_name)

    ## get the patch id for the changed patches
    def set_update_patch_list(self):

        if self.args.update:
            update_path = f'{self.args.dataset_path}/change_info.txt'
            self.update_patch_list = np.loadtxt(update_path, dtype = int)

            try:
                self.update_patch_list = list(self.update_patch_list)
            except:
                self.update_patch_list = [int(self.update_patch_list)]
        else:
            self.update_patch_list = None

    ## load the patch boxes number for each patch to construct the feature volumes
    def load_adaptive_grids(self, new_shape_name = None):

        ## load the adaptive grids information for the updated shape
        if not (new_shape_name is None):
            follow_shape_path = self.args.dataset_path.split('/')[:-1]
            follow_shape_path = '/'.join(follow_shape_path) + f'/{new_shape_name}'
            adaptive_path = f'{follow_shape_path}/adaptive_resolution.json'
            self.adaptive_patch_resolution_follow = read_json(adaptive_path)
            self.adaptive_patch_resolution_follow = {int(key): value for key, value in self.adaptive_patch_resolution_follow.items()}

        ## load the adaptive grids inforamtion for the original shape
        else:
            adaptive_path = f'{self.args.dataset_path}/adaptive_resolution.json'
            self.adaptive_patch_resolution = read_json(adaptive_path)
            self.adaptive_patch_resolution = {int(key): value for key, value in self.adaptive_patch_resolution.items()}

    def set_lr_scheduler(self):

        if self.args.lr_scheduler == 'exp':
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.args.lr_exp_gamma)

        elif self.args.lr_scheduler == 'multi_step':
            self.decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.decoder_opt, milestones = self.args.mile_stone, gamma = self.args.lr_multi_gamma)
            self.latent_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.latent_opt, milestones = self.args.mile_stone, gamma = self.args.lr_multi_gamma)

    ## get information for shape space
    def get_num_latents(self):

        latent_num_path = self.args.pretrained.replace('.pth','_latent.txt')
        latent_num = np.loadtxt(latent_num_path, dtype = int)

        return latent_num


    def set_dataset(self):

        self.dataset = SurfaceDataset(path = self.args.dataset_path, args = self.args, update_patch_id_list = self.update_patch_list, adaptive_patch_grids = self.adaptive_patch_resolution)    

        log.info("Loaded mesh dataset")
            
    def set_network(self):

        p_num = self.dataset.num_patches

        ## use shape space
        if self.args.use_shape_space:
            load_num_latents = self.get_num_latents() ## the patch num for training the shape space
            self.net = OctreeSDF(self.args, load_num_latents = load_num_latents, num_patches = p_num, device = self.device, adaptive_patch_resolution = self.adaptive_patch_resolution)
            self.net.load_state_dict(torch.load(self.args.pretrained))
            self.net.reset_latent_codes()
        ## ignore shape space
        else:
            if self.args.update:
                self.net = OctreeSDF(self.args, num_patches = p_num, device = self.device, adaptive_patch_resolution = self.adaptive_patch_resolution, follow_shape_name = self.args.follow_shape_name, follow_patch_resolution = self.adaptive_patch_resolution_follow)
            else:
                self.net = OctreeSDF(self.args, num_patches = p_num, device = self.device, adaptive_patch_resolution = self.adaptive_patch_resolution)

            if self.args.pretrained:
                self.net.load_state_dict(torch.load(self.args.pretrained))

            ## reset the latent codes index for the updated shape
            if self.args.update:
                self.net.reset_index_map()
                
        self.net.float().to(self.device)

        ## prepare information for merge loss normalization
        self.patch_res_arr = self.net.patch_res_arr
        
        ## get the map from points to feature volume id
        self.patch_grid_dict_map = map_patch_grid_dict(self.net.patch_grid_dict, self.net.max_code_res, self.net.num_patches)
        log.info("Total number of parameters: {}".format(sum(p.numel() for p in self.net.parameters())))

    def set_optimizer(self):

        self.latent_opt = optim.Adam(self.net.latent_codebook.parameters(), lr = self.args.latent_lr)
        self.decoder_opt = optim.Adam(self.net.pipeline.parameters(), lr = self.args.decoder_lr)


    def set_logger(self):

        if self.args.exp_name:
            self.log_fname = self.args.exp_name
        else:
            self.log_fname = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.log_dir = os.path.join(self.args.logs, self.log_fname)
        self.writer = SummaryWriter(self.log_dir, purge_step=0)
        self.writer.add_text('Parameters', self.args_str)

        log.info('Model configured and ready to go')


    def pre_epoch(self, epoch):

        self.net.train()

        # Initialize the dict for logging
        self.log_dict['total_loss'] = 0
        self.log_dict['patch_loss'] = 0
        self.log_dict['merge_loss'] = 0
        self.log_dict['total_iter_count'] = 0


    def iterate(self, epoch):
        
        data = self.dataset.get_data()
        self.step_geometry(epoch, data)


    def step_geometry(self, epoch, data):
        
        ## surface data
        surf_pts = data['points'].to(self.device).float()
        gt_surf_normals = data['normals'].to(self.device)
        pids = data['pids'].to(self.device)

        ## sdf data
        surf_jitter_pts = data['jitter_points'].to(self.device).float()
        gt_surf_jitter_sdfs = data['jitter_sdfs'].to(self.device)
        jitter_pids = data['jitter_pids'].to(self.device)

        ## spatial data
        off_surf_data =self.dataset.get_random_spatial_points()
        off_surf_pts = off_surf_data['spatial_points'].to(self.device).float()
        off_surf_pids = off_surf_data['spatial_pids'].to(self.device)

        ## merge data
        merge_data = self.dataset.get_random_merge_points()
        to_merge_surf_pts = merge_data['to_merge_surf_pts'].to(self.device).float()
        to_merge_query_pids = merge_data['to_merge_query_pids'].to(self.device)
        to_merge_surf_operator = merge_data['to_merge_surf_operator'].to(self.device)
        to_merge_surf_patch_pair = merge_data['to_merge_surf_patch_pair'].to(self.device)
        to_merge_ori_pids = merge_data['to_merge_ori_pids'].to(self.device)
        to_merge_smooth_mask = None
        to_merge_max_merge_count = merge_data['to_merge_max_merge_count']
        to_merge_max_patch_count = merge_data['to_merge_max_patch_count']

        # prepare for inference
        self.decoder_opt.zero_grad()
        self.latent_opt.zero_grad()
        
        # surf
        pred_sdfs, pred_normals = self.net(surf_pts, pids)

        # spatial 
        pred_sdfs_spatial, pred_normals_spatial = self.net(off_surf_pts, off_surf_pids)
        
        # sdf 
        pred_sdfs_jitter = self.net.forward_nograd(surf_jitter_pts, jitter_pids)

        ## compute patch loss
        self.patch_loss_dict = compute_patch_loss(
            self.args,
            pred_sdfs,
            pred_normals,
            gt_surf_normals,
            pred_normals_spatial, 
            pred_sdfs_jitter,
            gt_surf_jitter_sdfs)

        ## compute merge loss
        pred_sdfs_merge_list = self.args.impossible_num * torch.ones((to_merge_surf_pts.shape[0], to_merge_max_patch_count)).to(self.device).float()

        for i in range(to_merge_max_patch_count):

            ## get the mask for useful points
            query_mask = (to_merge_query_pids[:, i] != self.args.impossible_num)
            
            ## filter the points
            to_merge_query_patch_ids = to_merge_query_pids[query_mask, i]
            to_merge_surf_pts_piece = to_merge_surf_pts[query_mask]
            
            ## get the sdf prediction
            pred_sdfs_merge = self.net.forward_nograd(to_merge_surf_pts_piece, to_merge_query_patch_ids)
            pred_sdfs_merge_list[query_mask, i] = pred_sdfs_merge    

        self.merge_loss = compute_merge_loss(self.args, pred_sdfs_merge_list, \
                                             to_merge_surf_operator, to_merge_surf_patch_pair, to_merge_max_merge_count, \
                                             to_merge_max_patch_count, self.args.impossible_num, to_merge_ori_pids, to_merge_query_pids, \
                                             to_merge_surf_pts, self.patch_res_arr, self.patch_grid_dict_map, epoch, \
                                             self.update_patch_list, to_merge_smooth_mask
                                             )

        self.loss = self.patch_loss_dict['patch_loss'] 
        self.loss = self.loss + self.merge_loss['merge_loss']       

        self.loss = self.loss 
        self.loss.backward()
        self.latent_opt.step()
        self.decoder_opt.step()
        self.latent_lr_scheduler.step()
        self.decoder_lr_scheduler.step()

    ## update the loss log
    def update_loss_log(self):

        # patch loss 
        self.patch_loss = self.patch_loss_dict['patch_loss']
        self.surf_loss = self.patch_loss_dict['surf_loss']
        self.norm_loss = self.patch_loss_dict['norm_loss']
        self.eikonal_loss = self.patch_loss_dict['eikonal_loss']
        self.jitter_sdf_loss = self.patch_loss_dict['jitter_sdf_loss']
        
        self.log_dict['patch_loss'] = self.patch_loss.item()
        self.log_dict['surf_loss'] = self.surf_loss.item()
        self.log_dict['norm_loss'] = self.norm_loss.item()
        self.log_dict['spatial_loss'] = self.spatial_loss.item()
        self.log_dict['eikonal_loss'] = self.eikonal_loss.item()
        self.log_dict['jitter_sdf_loss'] = self.jitter_sdf_loss.item()

        ## merge loss
        self.log_dict['merge_loss'] = self.merge_loss

        self.log_dict['total_loss'] = self.loss.item()
        self.log_dict['total_iter_count'] = 1
    
    #######################
    # post_epoch
    #######################
    
    def post_epoch(self, epoch):

        self.net.eval()

        if epoch % self.args.save_every == 0:
            self.save_model(epoch)
        
        if epoch == self.args.epochs:
            print('Save the final epoch.')
            self.save_model(epoch)

        if epoch % self.args.print_loss_every == 0:
            self.print_loss(epoch)
            
    
    def print_loss(self, epoch):

        print_line = f'Epoch {epoch} | '
        for k, v in self.patch_loss_dict.items():
            print_line += f'{k}: {v:.4f} | '
        for k, v in self.merge_loss.items():
            print_line += f'{k}: {v:.4f} | '
        print_line += '\r'

        print(print_line)

    def save_model(self, epoch):

        log_comps = self.log_fname.split('/')
        if len(log_comps) > 1:
            _path = os.path.join(self.args.model_path, *log_comps[:-1])
            if not os.path.exists(_path):
                os.makedirs(_path)

        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

        if self.args.save_as_new:
            save_model_path = os.path.join(self.args.model_path,self.log_fname)
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            model_fname = os.path.join(save_model_path, f'{self.log_fname}_{epoch}.pth')
        else:
            model_fname = os.path.join(self.args.model_path, f'{self.log_fname}.pth')
        
        log.info(f'Saving model checkpoint to: {model_fname}')

        if self.args.save_all:
            torch.save(self.net, model_fname)
        else:
            torch.save(self.net.state_dict(), model_fname)

    def train(self):

        for epoch in tqdm(range(self.args.epochs)):    

            self.pre_epoch(epoch)
            self.iterate(epoch)
            self.post_epoch(epoch)

        self.writer.close()
    