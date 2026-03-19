import torch
import trimesh
import numpy as np 
from utils import return_box_cutted_mesh, judge_pts_in_box, get_num_patches
import json


class ShapeSpaceDataset():
    def __init__(self, path, args, adaptive_patch_grids = None):

        self.args = args

        self.shape_list = self.args.shape_list
        self.dataset_prefix = self.args.dataset_prefix

        self.num_patches_dict = self.get_shape_list_patch_num()
        ## get the total number of patches
        self.full_num_patches = sum(self.num_patches_dict.values())
        print(f'full_num_patches: {self.full_num_patches}')
        ## TODO: could also run the version of fixed resolution
        self.adaptive_patch_grids = adaptive_patch_grids

        ## load patches
        self.patch_mesh_dict = self.get_patch_meshes()
        self.new_patch_resolution_dict()

        ## if we use equal number of surface points in each box
        self.get_box_num_for_each_patch() 
        self.sample_points_on_surface_equal_box()
        self.sample_points_in_space()

        ## save the shape space map to json file
        with open(f'{self.args.model_path}/{self.args.exp_name}.json', 'w') as f:
            json.dump(self.shape_space_map, f)

    def new_patch_resolution_dict(self):
        self.adaptive_patch_resolution = {}
        patch_count = 0
        for shape_name in self.shape_list:
            num_patches = self.num_patches_dict[shape_name]
            for i in range(num_patches):
                resolution_cur = self.adaptive_patch_grids[shape_name][i]
                self.adaptive_patch_resolution[patch_count] = resolution_cur
                patch_count += 1

    def get_shape_list_patch_num(self):
        num_patches_dict = {}
        for shape_name in self.shape_list:
            shape_data_path = f'{self.args.dataset_prefix}/{shape_name}'
            num_patches = get_num_patches(shape_data_path)
            num_patches_dict[shape_name] = num_patches
        return num_patches_dict

    def get_patch_meshes(self):

        patch_mesh_dict = {}
        self.shape_space_map = {}
        patch_count = 0 
        for shape_name in self.shape_list:
            num_patches = self.num_patches_dict[shape_name]
            shape_data_path = f'{self.args.dataset_prefix}/{shape_name}'
            patch_mesh_dict[shape_name] = []
            for i in range(num_patches):
                self.shape_space_map[patch_count] = {'shape_id': shape_name, 'patch_id': i}
                patch_path = f'{shape_data_path}/patches/{i}.obj'    
                patch_mesh = trimesh.load(patch_path, process = False, maintain_order = True)            
                patch_mesh_dict[shape_name].append(patch_mesh)
                patch_count += 1

        return patch_mesh_dict
        
    ## get the bid list for each patch
    ## later used in octree grids
    def get_box_list_for_patch(self):
        patch_bid_dict = {}
        for merge_item in self.merge_order:
            for pid in merge_item['patch_ids']:
                if pid in patch_bid_dict.keys():
                    for bid in merge_item['box_id']:
                        patch_bid_dict[pid].append(bid)
                        patch_bid_dict[pid] = list(set(patch_bid_dict)) ## remove the redundant box id
                else:
                    patch_bid_dict[pid] = merge_item['box_id']

        return 
    
    def sample_jitter_surface(self, pts, normals,box_length):

        cur_jitter_points_out = pts + self.args.jitter_delta_scale * box_length * normals
        cur_jitter_points_in = pts - self.args.jitter_delta_scale * box_length * normals
        cur_jitter_points = np.concatenate((cur_jitter_points_out,cur_jitter_points_in), axis = 0)
  
        cur_jitter_normals_out = normals.copy()
        cur_jitter_normals_in = normals.copy()
        cur_jitter_normals = np.concatenate((cur_jitter_normals_out,cur_jitter_normals_in), axis = 0)

        cur_jitter_sdf_out = np.ones((cur_jitter_points_out.shape[0],1)) * self.args.jitter_delta_scale * box_length
        cur_jitter_sdf_in = np.ones((cur_jitter_points_in.shape[0],1)) * self.args.jitter_delta_scale * box_length * (-1)
        cur_jitter_sdfs = np.concatenate((cur_jitter_sdf_out,cur_jitter_sdf_in),axis = 0)
        
        cur_jitter_flag, cur_jitter_points = self.filter_off_samples(cur_jitter_points,return_flag = True)
        cur_jitter_normals = cur_jitter_normals[cur_jitter_flag]
        cur_jitter_sdfs = cur_jitter_sdfs[cur_jitter_flag]

        return cur_jitter_points, cur_jitter_normals, cur_jitter_sdfs

    def sample_points_on_surface_equal_box(self):

        self.all_list_points = []
        self.all_list_normals = []
        self.all_list_box_id = [] ## this is the box id for the feature volume
        self.all_list_jitter_points = []
        self.all_list_jitter_normals = []
        self.all_list_jitter_sdfs = []
        self.all_list_pid = []
        self.all_list_jitter_pid = []
        self.per_patch_per_fvbox_area = {}
        self.all_list_smooth_flag = []

        for pid in self.shape_space_map.keys():
            shape_name = self.shape_space_map[pid]['shape_id']
            chart_id = self.shape_space_map[pid]['patch_id']

            cur_patch_resolution = self.adaptive_patch_grids[shape_name][chart_id]
            box_length = 2 / 2**cur_patch_resolution 

            self.cur_patch_points = [] 
            self.cur_patch_normal = []
            self.cur_patch_jitter_points = []
            self.cur_patch_jitter_normals = []
            self.cur_patch_jitter_sdfs = []
            self.per_patch_per_fvbox_area[chart_id] = {}

            self.cur_box_id = []
            cur_box_info = self.box_info_surface_dict[shape_name][chart_id]
            cur_mesh = self.patch_mesh_dict[shape_name][chart_id]

            for i in range(cur_box_info.shape[0]):
                box_start_point = cur_box_info[i]
                if self.args.add_box_extension:
                    new_box_length = self.args.box_extension_scale * box_length
                    add_length = new_box_length - box_length
                    box_start_point = box_start_point - (add_length / 2)
                    cutted_mesh = return_box_cutted_mesh(cur_mesh,box_start_point,new_box_length)
                else:
                    cutted_mesh = return_box_cutted_mesh(cur_mesh,box_start_point,box_length)
                cutted_area = cutted_mesh.area
                self.per_patch_per_fvbox_area[chart_id][i] = cutted_area 
                points, fids = trimesh.sample.sample_surface(cutted_mesh, self.args.box_surf_num)
                normals = cutted_mesh.face_normals[fids]
                
                self.cur_patch_points.append(torch.FloatTensor(points))
                self.cur_patch_normal.append(torch.FloatTensor(normals))
                self.cur_box_id.append(torch.ones((points.shape[0],1)) * i)

                jitter_points, jitter_normals,jitter_sdfs = self.sample_jitter_surface(points, normals,box_length)
                self.cur_patch_jitter_points.append(torch.FloatTensor(jitter_points.copy()))
                self.cur_patch_jitter_normals.append(torch.FloatTensor(jitter_normals.copy()))
                self.cur_patch_jitter_sdfs.append(torch.FloatTensor(jitter_sdfs.copy()))

            self.cur_patch_points = torch.cat(self.cur_patch_points)
            self.cur_patch_normal = torch.cat(self.cur_patch_normal)

            self.cur_patch_jitter_points = torch.cat(self.cur_patch_jitter_points)
            self.cur_patch_jitter_normals = torch.cat(self.cur_patch_jitter_normals)
            self.cur_patch_jitter_sdfs = torch.cat(self.cur_patch_jitter_sdfs)
            self.cur_jitter_pid = torch.ones((self.cur_patch_jitter_points.shape[0],1)) * pid

            self.cur_box_id = torch.cat(self.cur_box_id)
            self.cur_pid = torch.ones((self.cur_patch_points.shape[0],1)) * pid 

            self.all_list_points.append(self.cur_patch_points)
            self.all_list_normals.append(self.cur_patch_normal)
            self.all_list_box_id.append(self.cur_box_id)
            self.all_list_pid.append(self.cur_pid)
            
            self.all_list_jitter_points.append(self.cur_patch_jitter_points)
            self.all_list_jitter_normals.append(self.cur_patch_jitter_normals)
            self.all_list_jitter_sdfs.append(self.cur_patch_jitter_sdfs)
            self.all_list_jitter_pid.append(self.cur_jitter_pid)

        self.all_list_points = torch.cat(self.all_list_points)
        self.all_list_normals = torch.cat(self.all_list_normals)
        self.all_list_box_id = torch.cat(self.all_list_box_id)
        self.all_list_pid = torch.cat(self.all_list_pid)
        self.all_list_smooth_flag = torch.zeros((self.all_list_points.shape[0],1))

        self.all_list_jitter_points = torch.cat(self.all_list_jitter_points)
        self.all_list_jitter_normals = torch.cat(self.all_list_jitter_normals)
        self.all_list_jitter_sdfs = torch.cat(self.all_list_jitter_sdfs)
        self.all_list_jitter_pid = torch.cat(self.all_list_jitter_pid)
    
    def get_box_num_for_each_patch(self):

        self.patch_box_num_list_surface_dict = {}
        self.box_info_surface_dict = {}
        self.patch_box_num_list_spatial_dict = {}
        self.box_info_spatial_dict = {}

        for shape_name in self.shape_list:

            num_p = self.num_patches_dict[shape_name]
            self.patch_box_num_list_surface_dict[shape_name] = []
            self.box_info_surface_dict[shape_name] = []
            self.patch_box_num_list_spatial_dict[shape_name] = []
            self.box_info_spatial_dict[shape_name] = []

            for chart_id in range(num_p):

                cur_patch_reso = 2**self.adaptive_patch_grids[shape_name][chart_id]
        
                box_path_spatial = f'{self.args.dataset_prefix}/{shape_name}/bounds/box_p{chart_id}_r{cur_patch_reso}_surface.bin'
                bounding_boxes_spatial = np.fromfile(box_path_spatial).reshape(-1,3)
                self.box_info_spatial_dict[shape_name].append(bounding_boxes_spatial)
                self.patch_box_num_list_spatial_dict[shape_name].append(bounding_boxes_spatial.shape[0])

                box_path_surface = f'{self.args.dataset_prefix}/{shape_name}/bounds/box_p{chart_id}_r{cur_patch_reso}_surface.bin'
                bounding_boxes_surface = np.fromfile(box_path_surface).reshape(-1,3)
                self.box_info_surface_dict[shape_name].append(bounding_boxes_surface)
                self.patch_box_num_list_surface_dict[shape_name].append(bounding_boxes_surface.shape[0])
        
    def sample_points_in_space(self):

        ## get off surface points 
        ## only sample points inside the redbox
        self.off_surface_pts_list = []
        self.off_surface_pid_list = []
        
        for pid in self.shape_space_map.keys():
            shape_name = self.shape_space_map[pid]['shape_id']
            chart_id = self.shape_space_map[pid]['patch_id']

            cur_patch_resolution = 2**self.adaptive_patch_grids[shape_name][chart_id]
    
            bounding_boxes = self.box_info_surface_dict[shape_name][chart_id]
            box_length = 2 / (cur_patch_resolution)

            off_samples = []

            for i in range(bounding_boxes.shape[0]):

                start_p = bounding_boxes[i]
                if self.args.add_box_extension:
                    new_box_length = self.args.box_extension_scale * box_length
                    add_length = new_box_length - box_length
                    start_p = start_p - (add_length / 2)
                    box_pts = self.sample_box_points(start_p,new_box_length)
                else:
                    box_pts = self.sample_box_points(start_p,box_length)

                if self.args.add_box_boundary:
                    box_pts_add = self.ori_box_boundary * box_length + start_p
                    off_samples.append(box_pts_add)
                off_samples.append(box_pts)

            off_samples = np.concatenate(off_samples, axis=0)
            off_samples = self.filter_off_samples(off_samples)
            self.off_surface_pts_list.append(torch.from_numpy(off_samples).float())
            self.off_surface_pid_list.append((torch.ones((off_samples.shape[0],1)) * pid).float())
        
        self.off_surface_pts_list = torch.cat(self.off_surface_pts_list)
        self.off_surface_pid_list = torch.cat(self.off_surface_pid_list)

    def sample_box_points(self,start_point,box_length):
        sample_num = self.args.box_uniform_num
        uniform_samples = np.random.uniform(start_point,start_point+box_length,(sample_num,3))
        return uniform_samples

    def filter_off_samples(self,off_samples,return_flag = False):
        
        start_p = np.array([-1,-1,-1])
        box_length = 2
        filter_flag = judge_pts_in_box(start_p,box_length,off_samples,remain_every_info=True, no_border = True)
        filtered_pts = off_samples[filter_flag]
        if return_flag:
            return filter_flag, filtered_pts
        else:
            return filtered_pts

    
    def get_data(self):

        if self.args.jitter_surface:
            jitter_rand_idx = np.random.choice(self.all_list_jitter_points.shape[0], size = self.args.batch_jitter_surf_num).reshape(-1)
        rand_idx = np.random.choice(self.all_list_points.shape[0], size = self.args.batch_surf_num).reshape(-1)

        pts_piece = self.all_list_points[rand_idx].reshape(-1,3)
        nor_piece = self.all_list_normals[rand_idx].reshape(-1,3)
        pid_piece = self.all_list_pid[rand_idx].reshape(-1)
        smooth_flag_piece = self.all_list_smooth_flag[rand_idx].reshape(-1)

        if self.args.jitter_surface:
            jitter_pts_piece = self.all_list_jitter_points[jitter_rand_idx].reshape(-1,3)
            jitter_nor_piece = self.all_list_jitter_normals[jitter_rand_idx].reshape(-1,3)
            jitter_sdf_piece = self.all_list_jitter_sdfs[jitter_rand_idx].reshape(-1)
            jitter_pid_piece = self.all_list_jitter_pid[jitter_rand_idx].reshape(-1)

        self.list_points = pts_piece 
        self.list_normals = nor_piece
        self.list_pids = pid_piece
        self.list_smooth_flag = smooth_flag_piece
        
        self.list_jitter_points = jitter_pts_piece 
        self.list_jitter_normals = jitter_nor_piece 
        self.list_jitter_sdfs = jitter_sdf_piece
        self.list_jitter_pids = jitter_pid_piece
        
        data = {
            'points': self.list_points,
            'normals': self.list_normals,
            'pids': self.list_pids.to(torch.long),
            'jitter_points': self.list_jitter_points,
            'jitter_normals': self.list_jitter_normals,
            'jitter_sdfs': self.list_jitter_sdfs,
            'jitter_pids': self.list_jitter_pids.to(torch.long)
        }

        return data


    def get_random_spatial_points(self):
        spatial_points = []
        rand_idx = np.random.choice(self.off_surface_pts_list.shape[0], size = self.args.batch_spatial_num).reshape(-1)
        spatial_points = self.off_surface_pts_list[rand_idx].reshape(-1,3)
        spatial_points_pid = self.off_surface_pid_list[rand_idx].reshape(-1)
        return {
            'spatial_points':spatial_points,
            'spatial_pids':spatial_points_pid.to(torch.long)
        }

