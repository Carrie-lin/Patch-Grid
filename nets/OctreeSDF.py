import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
from utils import get_num_patches

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def last_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)


## to compute the gradient of the output with respect to the input
def grad_compute(inputs, outputs):

        d_points = torch.ones_like(outputs, requires_grad=False, device=inputs.device)
        ori_grad = grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=d_points,  
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=False
        )
        points_grad = ori_grad[0]
        return points_grad


class OctreeSDF(nn.Module):
    def __init__(self,args,num_patches,device, load_num_latents = None, adaptive_patch_resolution = None, follow_shape_name = None, follow_patch_resolution = None, shape_space_map = None):

        super().__init__()
        self.code_res = args.code_res
        self.device = device
        self.num_patches = num_patches
        self.args = args
        if self.args.dataset_path is not None:
            shape_name = self.args.dataset_path.split('/')[-1]
            self.root = self.args.dataset_path.replace(f'/{shape_name}', '')

        self.follow_shape_name = follow_shape_name
        self.shape_space_map = shape_space_map

        if self.args.update:
            ## load the number of patches got from the shape to be followed
            follow_path = f'{self.root}/{self.follow_shape_name}'
            self.follow_num_patches = get_num_patches(follow_path)

        ## for updating the shape
        if self.args.update:
            change_info_path = f'{self.args.dataset_path}/change_info.txt'
            self.change_pid_list = np.loadtxt(change_info_path).tolist()

            try:
                self.change_pid_list = [int(item) for item in self.change_pid_list]
            except:
                self.change_pid_list = [int(self.change_pid_list)]

            self.adaptive_patch_resolution_follow = follow_patch_resolution
            self.adaptive_patch_resolution = adaptive_patch_resolution

            self.get_max_code_res()
            self.get_max_code_res(follow = True)

            self.load_grids_info()
            self.load_grids_info(follow = True)

            self.get_map_index(follow = True) ## construct the latent codes acording to the follow shape 
            self.get_map_index()
            self.build_codes_cpu()
            self.latent_codebook = nn.Embedding(self.follow_latent_count, args.emb_deg)
            
        ## for fitting a shape 
        else:
            self.adaptive_patch_resolution = adaptive_patch_resolution
            self.get_max_code_res()
            self.load_grids_info()
            self.get_map_index()
            self.build_codes_cpu()
            if self.args.use_shape_space:
                self.latent_codebook = nn.Embedding(load_num_latents, args.emb_deg) 
            else:
                self.latent_codebook = nn.Embedding(self.latent_count, args.emb_deg)
                self.set_latent_init()

        self.pipeline = nn.Sequential()

        for i in range(args.num_layers):
            if i == 0:
                self.pipeline.add_module(f'fc_{i}', nn.Linear(args.emb_deg, args.hidden_dim))
            else:
                self.pipeline.add_module(f'fc_{i}', nn.Linear(args.hidden_dim, args.hidden_dim))
            
            ## initialize the activation 
            if args.activation == 'sp':
                self.pipeline.add_module(f'act_{i}', nn.Softplus(beta=100))
            elif args.activation == 'relu':
                self.pipeline.add_module(f'act_{i}', nn.ReLU())
            elif args.activation == 'sine':
                self.pipeline.add_module(f'act_{i}', Sine())
                lin = getattr(self.pipeline, "fc_" + str(i))
                if i == 0:
                    first_layer_sine_init(lin)
                else:
                    sine_init(lin)
            else:
                print('Activation not implemented')
                raise NotImplementedError
    
        self.pipeline.add_module('fc_out', nn.Linear(args.hidden_dim, 1))
        if args.activation == 'sine':
            lin = getattr(self.pipeline, "fc_out")
            last_layer_sine_init(lin)


    ## if it's trained with shape space, reset the latent codes
    def reset_latent_codes(self):
        self.latent_codebook = nn.Embedding(self.latent_count, self.args.emb_deg)
        self.set_latent_init()

    def set_latent_init(self):
        self.latent_codebook.weight.data = torch.randn(self.latent_count, self.args.emb_deg).float().to(self.device)/self.args.emb_deg 
   
    def generate_index_map(self,follow = False):
        if follow:
            grid_nums = self.grid_num_dict_follow
        else:
            grid_nums = self.grid_num_dict
        
        current_count = 0
        index_map = {}

        for pid in grid_nums.keys():
            grid_num = grid_nums[pid]
            current_count += grid_num 
            index_map[pid] = current_count  
        
        return index_map

    ## copy the latent codes from original map to current map
    def reset_index_map(self):
        ## get the patch index

        ## new a set of latent codes
        new_latent = nn.Embedding(self.latent_count, self.args.emb_deg)
        new_latent.weight.data = torch.randn(self.latent_count, self.args.emb_deg).float().to(self.device)/self.args.emb_deg 

        ## copy previous latent codes
        new_latent = self.copy_codes(new_latent, self.now_map, self.follow_map)
        self.latent_codebook = new_latent

        ## reset the map 
        self.build_codes_cpu()

        return 

    def copy_codes(self,new_latent,now_map,follow_map):

        for pid in range(self.num_patches):
            if not (pid in self.change_pid_list):
                if pid == 0:
                    now_p_start = 0
                    follow_p_start = 0
                else:
                    now_p_start = now_map[pid-1]
                    follow_p_start = follow_map[pid-1]
                now_p_end = now_map[pid]
                follow_p_end = follow_map[pid]
                new_latent.weight.data[now_p_start:now_p_end,:] = self.latent_codebook.weight.data[follow_p_start:follow_p_end,:]
        return new_latent

    def get_map_index(self, follow = False):   

        ## construct patch grid dict
        patch_grid_dict = {}
        if follow:
            patch_num_cur = self.follow_num_patches
        else:
            patch_num_cur = self.num_patches

        for pid in range(patch_num_cur):
            patch_grid_dict[pid] = []
            if follow:
                patch_reso = self.adaptive_patch_resolution_follow[pid]
            else:
                patch_reso = self.adaptive_patch_resolution[pid]
 
            box_len = 2 / 2**patch_reso

            if follow:
                for gid in range(self.grid_info_dict_follow[pid].shape[0]):
      
                    sp = self.grid_info_dict_follow[pid][gid]

                    i = int((sp[0]+1)/box_len)
                    j = int((sp[1]+1)/box_len)
                    k = int((sp[2]+1)/box_len)

                    patch_grid_dict[pid].extend([[i,j,k],[i,j,k+1],[i,j+1,k],[i+1,j,k],\
                        [i,j+1,k+1],[i+1,j+1,k],[i+1,j,k+1],[i+1,j+1,k+1]])

                patch_grid_dict[pid] = self.remove_duplicate(patch_grid_dict[pid])

            else:

                for gid in range(self.grid_info_dict[pid].shape[0]):
                    sp = self.grid_info_dict[pid][gid]

                    i = int((sp[0]+1)/box_len)
                    j = int((sp[1]+1)/box_len)
                    k = int((sp[2]+1)/box_len)
                    
                    patch_grid_dict[pid].extend([[i,j,k],[i,j,k+1],[i,j+1,k],[i+1,j,k],\
                        [i,j+1,k+1],[i+1,j+1,k],[i+1,j,k+1],[i+1,j+1,k+1]])

                patch_grid_dict[pid] = self.remove_duplicate(patch_grid_dict[pid])

        self.patch_grid_dict = patch_grid_dict

        ## construct dict mapping
        self.ijk_map = {}
        
        if follow:
            self.follow_map = {}
            self.follow_latent_count = 0

            for p in range(self.follow_num_patches):
                for item in patch_grid_dict[p]:
                    self.follow_latent_count += 1
                self.follow_map[p] = self.follow_latent_count
        
        else:
            self.now_map = {}
            self.latent_count = 0

            for p in range(self.num_patches):

                for item in patch_grid_dict[p]:

                    new_pijk = str([p,item[0],item[1],item[2]])
                    self.ijk_map[new_pijk] = self.latent_count
                    self.latent_count += 1

                self.now_map[p] = self.latent_count

    def remove_duplicate(self, list2):

        new_list = []
        for item in list2:
            if item not in new_list:
                new_list.append(item)

        return new_list

    def load_grids_info(self,follow = False):

        ## for update
        if follow:
            grid_info_prefix = f'{self.root}/{self.follow_shape_name}'
            self.grid_num_dict_follow = {}
            self.grid_info_dict_follow = {}
            self.patch_res_arr_follow = np.zeros((self.follow_num_patches,1)).reshape(-1)
            for pid in range(self.follow_num_patches):
                
                cur_reso = 2**self.adaptive_patch_resolution_follow[pid]
                grid_path = f'{grid_info_prefix}/bounds/box_p{pid}_r{cur_reso}_extend.bin'
                grid_info = np.fromfile(grid_path).reshape(-1,3)
                grid_num = grid_info.shape[0]
                self.grid_num_dict_follow[pid] = grid_num
                self.grid_info_dict_follow[pid] = grid_info
                self.patch_res_arr_follow[pid] = cur_reso
            
            self.patch_res_arr_follow = torch.from_numpy(self.patch_res_arr_follow).float().to(self.device)

        ## for training the shape space
        elif self.args.train_shape_space:
            self.grid_num_dict = {}
            self.grid_info_dict = {}
            self.patch_res_arr = np.zeros((self.num_patches,1)).reshape(-1)
            for pid in range(self.num_patches):

                shape_name = self.shape_space_map[pid]['shape_id']
                chart_id = self.shape_space_map[pid]['patch_id']

                grid_info_prefix = f'{self.args.dataset_prefix}/{shape_name}'
    
  
                cur_reso = 2**self.adaptive_patch_resolution[pid]

                grid_path = f'{grid_info_prefix}/bounds/box_p{chart_id}_r{cur_reso}_extend.bin'
                grid_info = np.fromfile(grid_path).reshape(-1,3)
                grid_num = grid_info.shape[0]
                self.grid_num_dict[pid] = grid_num
                self.grid_info_dict[pid] = grid_info
                self.patch_res_arr[pid] = cur_reso
            self.patch_res_arr = torch.from_numpy(self.patch_res_arr).float().to(self.device)

        ## for training a single shape
        else:
            grid_info_prefix = self.args.dataset_path
            self.grid_num_dict = {}
            self.grid_info_dict = {}
            self.patch_res_arr = np.zeros((self.num_patches,1)).reshape(-1)

            for pid in range(self.num_patches):
                cur_reso = 2**self.adaptive_patch_resolution[pid]
                grid_path = f'{grid_info_prefix}/bounds/box_p{pid}_r{cur_reso}_extend.bin'
                grid_info = np.fromfile(grid_path).reshape(-1,3)
                grid_num = grid_info.shape[0]
                self.grid_num_dict[pid] = grid_num
                self.grid_info_dict[pid] = grid_info
                self.patch_res_arr[pid] = cur_reso
            self.patch_res_arr = torch.from_numpy(self.patch_res_arr).float().to(self.device)

    def freeze_part_latent(self):
  
        for pid in range(self.num_patches):
            ## if the patch is not edited, make it static
            if (pid in self.change_pid_list):
            
                if pid == 0:
                    now_p_start = 0
                else:
                    now_p_start = self.now_map[pid-1]
                now_p_end = self.now_map[pid]
                weight = self.latent_codebook.weight[now_p_start:now_p_end,:]
                weight.requires_grad = False

    def freeze_decoder(self):
        for k,v in self.pipeline.named_parameters():
            v.requires_grad = False

    def enable_decoder(self):
        for k,v in self.pipeline.named_parameters():
            v.requires_grad = True

    def build_codes_cpu(self):
        
        self.codes = (-1) * torch.ones(self.num_patches,self.max_code_res, self.max_code_res, self.max_code_res).long()
        cur_ijk_map = self.ijk_map

        for pijk_item in cur_ijk_map.keys():
            pijk_item_int = eval(pijk_item)
            pid = int(pijk_item_int[0])
            i = int(pijk_item_int[1])
            j = int(pijk_item_int[2])
            k = int(pijk_item_int[3])
            self.codes[pid, i, j, k] = cur_ijk_map[pijk_item]
        
    def get_max_code_res(self, follow=False):
        if follow:
            self.max_code_res_follow = 0
            for pid in self.adaptive_patch_resolution_follow:
                code_res = 2**self.adaptive_patch_resolution_follow[pid] + 1
                if code_res > self.max_code_res_follow:
                    self.max_code_res_follow = code_res
        else:
            self.max_code_res = 0
            for pid in self.adaptive_patch_resolution:
                code_res = 2**self.adaptive_patch_resolution[pid] + 1
                if code_res > self.max_code_res:
                    self.max_code_res = code_res
        
    def forward(self, x, pid):
        """
        x: [K, (N, 3)]
        """
        x.requires_grad = True
        x_pred_sdf = self.forward_nograd(x, pid)
        x_normals = grad_compute(x, x_pred_sdf)
        return x_pred_sdf, x_normals
    
    def forward_nograd(self, x, pid):
    
        l = self.query_codes(x, pid) # [5000, 256]

        x_pred_sdf = self.pipeline(l).squeeze()

        return x_pred_sdf

    def query_codes(self, x, mask):

        ## x is a float from -1 to 1
        x_ori = x.detach().cpu().numpy()
        x = (x + 1) / 2

        special_res = self.patch_res_arr[mask].reshape(-1,1)
        x = x * (special_res)

        xlong = x.long() 
        delta = x - xlong
      
        return self.special_trilinear_interp(xlong, delta, mask, x_ori)

    def special_trilinear_interp(self, xlong, delta, mask, x):

        delta = delta[..., None]
        xlong = xlong.detach().cpu().numpy()

        c000 = self.codes[mask, xlong[:, 0], xlong[:, 1], xlong[:, 2]].to(self.device).long()
        c100 = self.codes[mask, xlong[:, 0]+1, xlong[:, 1], xlong[:, 2]].to(self.device).long()
        
        c001 = self.codes[mask, xlong[:, 0], xlong[:, 1], xlong[:, 2]+1].to(self.device).long()
        c101 = self.codes[mask, xlong[:, 0]+1, xlong[:, 1], xlong[:, 2]+1].to(self.device).long()
        
        c010 = self.codes[mask, xlong[:, 0], xlong[:, 1]+1, xlong[:, 2]].to(self.device).long()
        c110 = self.codes[mask, xlong[:, 0]+1, xlong[:, 1]+1, xlong[:, 2]].to(self.device).long()

        c011 = self.codes[mask, xlong[:, 0], xlong[:, 1]+1, xlong[:, 2]+1].to(self.device).long()
        c111 = self.codes[mask, xlong[:, 0]+1, xlong[:, 1]+1, xlong[:, 2]+1].to(self.device).long()

        self.detect_wrong_sample(c000, c100, c001, c101, c010, c110, c011, c111, mask, x)

        ## tri-linear interpolation
        l000 = self.latent_codebook.weight[c000]
        l100 = self.latent_codebook.weight[c100]

        l001 = self.latent_codebook.weight[c001]
        l101 = self.latent_codebook.weight[c101]

        l010 = self.latent_codebook.weight[c010]
        l110 = self.latent_codebook.weight[c110]

        l011 = self.latent_codebook.weight[c011]
        l111 = self.latent_codebook.weight[c111]

        l00 = l000*(1-delta[:, 0]) + l100*delta[:, 0]
        l01 = l001*(1-delta[:, 0]) + l101*delta[:, 0]
        l10 = l010*(1-delta[:, 0]) + l110*delta[:, 0]
        l11 = l011*(1-delta[:, 0]) + l111*delta[:, 0]

        l0 = l00*(1-delta[:, 1]) + l10*delta[:, 1]
        l1 = l01*(1-delta[:, 1]) + l11*delta[:, 1]

        l = l0*(1-delta[:, 2]) + l1*delta[:, 2]

        l = l.float()

        return l

    ## detect whether there are wrong samples
    ## that means there are points beyond the boundary of the feature volume
    def detect_wrong_sample(self, c000, c100, c001, c101, c010, c110, c011, c111, mask, x):

        variables = [c000, c100, c001, c101, c010, c110, c011, c111]
        names = ['c000', 'c100', 'c001', 'c101', 'c010', 'c110', 'c011', 'c111']

        for var, name in zip(variables, names):
            if torch.eq(var, -1).any():
                print('Sample points beyond the boundary!')
                exit(0)
        return 
    
    def embedding_reg_loss(self):
        reg_loss = 0.0
        weight = self.latent_codebook.weight.reshape(-1, self.args.emb_deg)
        reg_loss = torch.norm(weight, dim=1).mean()
        return reg_loss
    



