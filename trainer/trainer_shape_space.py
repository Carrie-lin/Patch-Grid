from datetime import datetime
import os

import logging as log
import torch
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from loss.patch_loss import compute_patch_loss_ss
import torch.optim.lr_scheduler
from dataset.dataset_shape_space import ShapeSpaceDataset
from nets.OctreeSDF import OctreeSDF
from utils import PerfTimer, read_json


class ShapeSpaceTrainer(object):

    def __init__(self, args, args_str):

        self.args = args 
        self.args_str = args_str
        
        self.args.epochs += 1

        self.shape_list = args.shape_list 

        self.timer = PerfTimer(activate = self.args.perf)
        self.timer.reset()

        # Set device to use
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        device_name = torch.cuda.get_device_name(device = self.device)
        log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        self.log_dict = {}

        # Initialize
        self.set_adapt_grids()
        self.set_dataset()
        self.set_network()
        self.set_optimizer()
        self.set_lr_scheduler()
        self.set_logger()

    ## get the feature volume grids number for each patch
    def set_adapt_grids(self):
        self.load_adaptive_grids()

    ## load the patch boxes number for each patch to construct the feature volumes
    def load_adaptive_grids(self):

        self.adaptive_patch_resolution = {}
        for shape_name in self.shape_list:
            dataset_path = f'{self.args.dataset_prefix}/{shape_name}'
            adaptive_path = f'{dataset_path}/adaptive_resolution.json'
            cur_shape_adaptive_patch_resolution = read_json(adaptive_path)
            self.adaptive_patch_resolution[shape_name] = {int(key): value for key, value in cur_shape_adaptive_patch_resolution.items()}

    def set_lr_scheduler(self):

        if self.args.lr_scheduler == 'exp':
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.args.lr_exp_gamma)

        elif self.args.lr_scheduler == 'multi_step':
            self.decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.decoder_opt, milestones = self.args.mile_stone, gamma = self.args.lr_multi_gamma)
            self.latent_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.latent_opt, milestones = self.args.mile_stone, gamma = self.args.lr_multi_gamma)

    def set_dataset(self):

        self.dataset = ShapeSpaceDataset(path = self.args.dataset_prefix, args= self.args, adaptive_patch_grids = self.adaptive_patch_resolution)    
        log.info("Loaded mesh dataset")
            
    def set_network(self):

        p_num = self.dataset.full_num_patches
        adaptive_patch_resolution = self.dataset.adaptive_patch_resolution
        shape_space_map = self.dataset.shape_space_map
        self.net = OctreeSDF(self.args, num_patches = p_num, device = self.device, adaptive_patch_resolution = adaptive_patch_resolution, shape_space_map = shape_space_map)
        
        ## save latent num as int
        self.latent_count = self.net.latent_count
        ## save one int number to txt
        save_path = f'{self.args.model_path}/{self.args.exp_name}_latent.txt'
        with open(save_path, 'w') as f:
            f.write(str(self.latent_count))
        f.close()
        #np.savetxt(f'{self.args.model_path}/{self.args.exp_name}_latent.txt', self.latent_count)  
       
        if self.args.pretrained:
            self.net.load_state_dict(torch.load(self.args.pretrained))

        self.net.to(self.device)

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


    def pre_epoch(self):

        self.net.train()

        # Initialize the dict for logging
        self.log_dict['total_loss'] = 0
        self.log_dict['patch_loss'] = 0
        self.log_dict['merge_loss'] = 0
        self.log_dict['total_iter_count'] = 0

    def iterate(self):

        data = self.dataset.get_data()
        self.step_geometry(data)


    def step_geometry(self, data):
        
        ## surface data
        surf_pts = data['points'].to(self.device)
        gt_surf_normals = data['normals'].to(self.device)
        pids = data['pids'].to(self.device)

        ## sdf data
        surf_jitter_pts = data['jitter_points'].to(self.device)
        gt_surf_jitter_sdfs = data['jitter_sdfs'].to(self.device)
        jitter_pids = data['jitter_pids'].to(self.device)

        ## spatial data
        off_surf_data =self.dataset.get_random_spatial_points()
        off_surf_pts = off_surf_data['spatial_points'].to(self.device)
        off_surf_pids = off_surf_data['spatial_pids'].to(self.device)

        # Prepare for inference
        self.decoder_opt.zero_grad()
        self.latent_opt.zero_grad()
        
        # Get prediction surf
        pred_sdfs, pred_normals = self.net(surf_pts.float(), pids)
    
        # spatial
        _, pred_normals_spatial = self.net(off_surf_pts, off_surf_pids)
        
        # sdf 
        pred_sdfs_jitter = self.net.forward_nograd(surf_jitter_pts, jitter_pids)

        ## patch loss
        self.patch_loss_dict = compute_patch_loss_ss(
            self.args,
            pred_sdfs,
            pred_normals,
            gt_surf_normals,
            pred_normals_spatial, 
            pred_sdfs_jitter,
            gt_surf_jitter_sdfs)
        
        self.loss = self.patch_loss_dict['patch_loss']

        ## backpropagate
        self.loss.backward()

        ## clear the gradient
        self.latent_opt.step()
        self.decoder_opt.step()
        self.latent_lr_scheduler.step()
        self.decoder_lr_scheduler.step()
    
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
            
        self.timer.check('post_epoch done')
    
    def print_loss(self, epoch):

        print_line = f'Epoch {epoch} | '
        for k, v in self.patch_loss_dict.items():
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

            self.pre_epoch()
            self.iterate()
            self.post_epoch(epoch)

        self.writer.close()
    