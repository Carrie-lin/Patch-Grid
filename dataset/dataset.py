import os
import torch
import trimesh
import glob
import numpy as np 
from preparation.lib.mergeutils import RedBox
from utils import read_json, get_num_patches, get_box_id_for_pts, filter_points_with_pids, filter_query_pids_with_update_pids
from dataset.sample_utils.sample_surface import sample_points_on_surface_equal_box
from dataset.sample_utils.sample_space import sample_space_pts
from dataset.sample_utils.sample_merge import presample_points_in_merge_cells, sample_surf_points_for_merge

class SurfaceDataset():
    def __init__(self, path, args, update_patch_id_list = None, adaptive_patch_grids = None):
        self.root = path
        self.args = args
     
        self.patch_resolution = self.args.code_res - 1

        paths = glob.glob(os.path.join(self.root, '*.obj'))
        paths = sorted(paths)

        self.num_patches = get_num_patches(self.root)
        self.adaptive_patch_grids = adaptive_patch_grids

        self.redboxes = []
        self.merge_order= {}
        self.load_merge_order()
        self.get_merge_redbox()
        self.load_open_bdrs()
        self.load_patches()
 
        ## judge whether the data has been sampled
        self.find_presampled_data()
        self.to_merge_smooth_mask = None

        if self.already_sample:
            self.load_presampled_data()
        else:
            self.get_box_num_for_each_patch() 
            self.sample_surface_pts()
            self.sample_space_pts()
            self.sample_merge_pts()
            self.save_presampled_data()
        
        ## for update case, filter the points only for updated patches
        if self.args.update:
            self.update_patch_id_list = update_patch_id_list 
            self.filter_involve_data()

    ## load open boundaries for shapes with open boundaries
    def load_open_bdrs(self):

        self.open_bdr_info_dict = {}
        self.open_bdr_distribution_dict = {}
        if self.args.open:
            bdr_path_template = f'{self.root}/bdr*.npy'
            bdr_paths = glob.glob(bdr_path_template)
            for bdr_id in range(len(bdr_paths)):
                bdr_path = f'{self.root}/bdr{bdr_id+1}.npy'
                bdr_info = np.load(bdr_path)
                bdr_info[:,3:] = -1 * bdr_info[:,3:]
                self.open_bdr_distribution_dict[bdr_id+1] = bdr_info.shape[0]
                self.open_bdr_info_dict[bdr_id+1] = bdr_info
    
    ## for the task of shape update, filter the training points only for updated patches   
    def filter_involve_data(self):
        
        ## filter surface points
        filter_flag = filter_points_with_pids(self.update_patch_id_list, self.surface_pids)
        self.surface_points = self.surface_points[filter_flag]
        self.surface_normals = self.surface_normals[filter_flag]
        self.surface_pids = self.surface_pids[filter_flag]
  
        ## filter sdf points
        filter_flag = filter_points_with_pids(self.update_patch_id_list, self.jitter_pids)
        self.jitter_points = self.jitter_points[filter_flag]
        self.jitter_normals = self.jitter_normals[filter_flag]
        self.jitter_sdfs = self.jitter_sdfs[filter_flag]
        self.jitter_pids = self.jitter_pids[filter_flag]

        ## filter merge points
        filter_flag = filter_query_pids_with_update_pids(self.to_merge_query_pids, self.update_patch_id_list)
        self.to_merge_surf_pts = self.to_merge_surf_pts[filter_flag]
        self.to_merge_query_pids = self.to_merge_query_pids[filter_flag]
        self.to_merge_ori_pids = self.to_merge_ori_pids[filter_flag]
        self.to_merge_surf_normals = self.to_merge_surf_normals[filter_flag]
        self.to_merge_surf_operator = self.to_merge_surf_operator[filter_flag]
        self.to_merge_surf_patch_pair = self.to_merge_surf_patch_pair[filter_flag]

        ## load spatial points
        filter_flag = filter_points_with_pids(self.update_patch_id_list, self.spatial_pids)
        self.spatial_points = self.spatial_points[filter_flag]
        self.spatial_pids = self.spatial_pids[filter_flag]

    def save_presampled_data(self):

        shape_name = (self.root).split('/')[-1]
        save_path = f'{self.args.dataset_save}/{shape_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        ## save surface points
        np.save(f'{save_path}/surface_points.npy', self.surface_points)
        np.save(f'{save_path}/surface_normals.npy', self.surface_normals)
        np.save(f'{save_path}/surface_pids.npy', self.surface_pids)

        ## save sdf points
        np.save(f'{save_path}/jitter_points.npy', self.jitter_points)
        np.save(f'{save_path}/jitter_normals.npy', self.jitter_normals)
        np.save(f'{save_path}/jitter_sdfs.npy', self.jitter_sdfs)
        np.save(f'{save_path}/jitter_pids.npy', self.jitter_pids)

        ## save merge points
        np.save(f'{save_path}/to_merge_surf_pts.npy', self.to_merge_surf_pts)
        np.save(f'{save_path}/to_merge_surf_normals.npy', self.to_merge_surf_normals)
        np.save(f'{save_path}/to_merge_query_pids.npy', self.to_merge_query_pids)
        np.save(f'{save_path}/to_merge_ori_pids.npy', self.to_merge_ori_pids)
        np.save(f'{save_path}/to_merge_surf_operator.npy', self.to_merge_surf_operator)
        np.save(f'{save_path}/to_merge_surf_patch_pair.npy', self.to_merge_surf_patch_pair)
        np.save(f'{save_path}/to_merge_max_merge_count.npy', self.to_merge_max_merge_count)
        np.save(f'{save_path}/to_merge_max_patch_count.npy', self.to_merge_max_patch_count)

        ## save spatial points
        np.save(f'{save_path}/spatial_points.npy', self.spatial_points)
        np.save(f'{save_path}/spatial_pids.npy', self.spatial_pids)

    ## judge whether the data has been sampled
    def find_presampled_data(self):

        self.already_sample = False 

        shape_name = (self.root).split('/')[-1]
        save_path = f'{self.args.dataset_save}/{shape_name}'
        
        if os.path.exists(save_path):
            self.already_sample = True
        
    def load_presampled_data(self):

        print('Load presampled data...')

        shape_name = (self.root).split('/')[-1]
        save_path = f'{self.args.dataset_save}/{shape_name}'

        ## load surface points
        self.surface_points = torch.from_numpy(np.load(f'{save_path}/surface_points.npy')).float()
        self.surface_normals = torch.from_numpy(np.load(f'{save_path}/surface_normals.npy')).float()
        self.surface_pids = torch.from_numpy(np.load(f'{save_path}/surface_pids.npy')).long()

        ## load jitter points
        self.jitter_points = torch.from_numpy(np.load(f'{save_path}/jitter_points.npy')).float()
        self.jitter_normals = torch.from_numpy(np.load(f'{save_path}/jitter_normals.npy')).float()
        self.jitter_sdfs = torch.from_numpy(np.load(f'{save_path}/jitter_sdfs.npy')).float()
        self.jitter_pids = torch.from_numpy(np.load(f'{save_path}/jitter_pids.npy')).long()

        ## load merge points
        self.to_merge_surf_pts = torch.from_numpy(np.load(f'{save_path}/to_merge_surf_pts.npy')).float()
        self.to_merge_query_pids = torch.from_numpy(np.load(f'{save_path}/to_merge_query_pids.npy')).long()
        self.to_merge_ori_pids = torch.from_numpy(np.load(f'{save_path}/to_merge_ori_pids.npy')).long()
        self.to_merge_surf_normals = torch.from_numpy(np.load(f'{save_path}/to_merge_surf_normals.npy')).float()
        self.to_merge_surf_operator = torch.from_numpy(np.load(f'{save_path}/to_merge_surf_operator.npy')).long()
        self.to_merge_surf_patch_pair = torch.from_numpy(np.load(f'{save_path}/to_merge_surf_patch_pair.npy')).long()
        self.to_merge_max_merge_count = torch.from_numpy(np.load(f'{save_path}/to_merge_max_merge_count.npy')).long()
        self.to_merge_max_patch_count = torch.from_numpy(np.load(f'{save_path}/to_merge_max_patch_count.npy')).long()
        try:
            self.to_merge_smooth_mask = torch.from_numpy(np.load(f'{save_path}/to_merge_smooth_mask.npy')).bool()
        except:
            self.to_merge_smooth_mask = None
        ## load spatial points
        self.spatial_points = torch.from_numpy(np.load(f'{save_path}/spatial_points.npy')).float()
        self.spatial_pids = torch.from_numpy(np.load(f'{save_path}/spatial_pids.npy')).long()
        
    def sample_merge_pts(self):

        print('Sample merge points...') 
        merge_surf_points, merge_surf_normals, merge_surf_pids, merge_surf_box_ids = sample_surf_points_for_merge(self.patch_meshes, self.merge_order, self.redboxes, self.args.box_surf_num, self.args.total_surf_num, self.patch_area_list, self.args.merge_extend_length, self.args.open)
        
        merge_data = presample_points_in_merge_cells(self.merge_order, merge_surf_points, merge_surf_normals, merge_surf_box_ids, merge_surf_pids, self.args.impossible_num)

        self.to_merge_ori_pids = merge_data['to_merge_ori_pids'].long()
        self.to_merge_surf_pts = merge_data['to_merge_surf_pts'].float()
        self.to_merge_query_pids = merge_data['to_merge_query_pids'].long()
        self.to_merge_surf_normals = merge_data['to_merge_surf_normals'].float()

        self.to_merge_surf_operator = merge_data['to_merge_surf_operator'].long()
        self.to_merge_surf_patch_pair = merge_data['to_merge_surf_patch_pair'].long()
        
        self.to_merge_max_merge_count = merge_data['to_merge_max_merge_count']
        self.to_merge_max_patch_count = merge_data['to_merge_max_patch_count']

    def sample_space_pts(self):

        print('Sample spatial points...')
        spatial_points = sample_space_pts(self.adaptive_patch_grids, self.args, self.patch_fv_list)
        
        self.spatial_points = spatial_points['spatial_points']
        self.spatial_pids = spatial_points['spatial_pids']

        self.spatial_box_ids = get_box_id_for_pts(self.spatial_points, self.redboxes)

    def sample_surface_pts(self):

        print('Sample surface points...')
        
        surface_pts, jitter_pts = sample_points_on_surface_equal_box(self.patch_meshes, self.patch_fv_list, self.adaptive_patch_grids, self.args.jitter_delta_scale, self.args.box_surf_num, self.open_bdr_info_dict)

        ## prepare jitter points
        self.jitter_points = jitter_pts['jitter_points']
        self.jitter_normals = jitter_pts['jitter_normals']
        self.jitter_sdfs = jitter_pts['jitter_sdfs']
        self.jitter_pids = jitter_pts['jitter_pids']

        ## prepare surface points
        self.surface_points = surface_pts['points']
        self.surface_normals = surface_pts['normals']
        self.surface_pids = surface_pts['pids']

    def load_patches(self):
        
        self.patch_meshes = []
        self.patch_area_list = []

        ## load patch list
        for i in range(self.num_patches):
            patch_path = f'{self.root}/patches/{i}.obj'    
            patch_mesh = trimesh.load(patch_path, process = False, maintain_order = True)      
            patch_area = patch_mesh.area      
            self.patch_meshes.append(patch_mesh)
            self.patch_area_list.append(patch_area)

    def get_box_num_for_each_patch(self):

        self.patch_fv_list = []
        self.patch_fv_num_list = []

        for chart_id in range(len(self.adaptive_patch_grids)):

            cur_patch_reso = 2**self.adaptive_patch_grids[chart_id]

            patch_fv_path = f'{self.root}/bounds/box_p{chart_id}_r{cur_patch_reso}_surface.bin'
            patch_fv = np.fromfile(patch_fv_path).reshape(-1,3)
            self.patch_fv_list.append(patch_fv)
            self.patch_fv_num_list.append(patch_fv.shape[0])
    
    def get_merge_redbox(self):

        file_redboxes = os.path.join(self.root,f'redboxes_adapt.json')
  
        if os.path.exists(file_redboxes):
            redboxes = read_json(file_redboxes)
            for rbx in redboxes:
                half_size = rbx['half_size'] 
                rbxx = RedBox(np.array(rbx['center']), half_size)
                rbxx.set_face_ids(rbx['face_ids'])
                rbxx.set_height(rbx['height'])
                self.redboxes.append(rbxx)
        
    def load_merge_order(self):

        merge_path = f'{self.root}/merge_order_adapt.json'
        self.merge_order = read_json(merge_path)
    

    def get_data(self):

        ## get surface data
        surf_rand_idx = np.random.choice(self.surface_points.shape[0], size = self.args.batch_surf_num).reshape(-1)

        pts_piece = self.surface_points[surf_rand_idx].reshape(-1,3)
        nor_piece = self.surface_normals[surf_rand_idx].reshape(-1,3)
        pid_piece = self.surface_pids[surf_rand_idx].reshape(-1)


        ## get sdf data
        jitter_rand_idx = np.random.choice(self.jitter_points.shape[0], size = self.args.batch_jitter_surf_num).reshape(-1)
    
        jitter_pts_piece = self.jitter_points[jitter_rand_idx].reshape(-1,3)
        jitter_nor_piece = self.jitter_normals[jitter_rand_idx].reshape(-1,3)
        jitter_sdf_piece = self.jitter_sdfs[jitter_rand_idx].reshape(-1)
        jitter_pid_piece = self.jitter_pids[jitter_rand_idx].reshape(-1)

        self.list_jitter_points = jitter_pts_piece 
        self.list_jitter_normals = jitter_nor_piece 
        self.list_jitter_sdfs = jitter_sdf_piece
        self.list_jitter_pids = jitter_pid_piece

        data = {
            'points': pts_piece,
            'normals': nor_piece,
            'pids': pid_piece.to(torch.long),
            'jitter_points': self.list_jitter_points,
            'jitter_normals': self.list_jitter_normals,
            'jitter_sdfs': self.list_jitter_sdfs,
            'jitter_pids': self.list_jitter_pids.to(torch.long),
        }

        return data

    def get_random_spatial_points(self):

        rand_idx = np.random.choice(self.spatial_points.shape[0], size = self.args.batch_spatial_num).reshape(-1)
        spatial_points = self.spatial_points[rand_idx].reshape(-1,3)
        spatial_points_pid = self.spatial_pids[rand_idx].reshape(-1)

        return {
            'spatial_points':spatial_points,
            'spatial_pids':spatial_points_pid.to(torch.long)
        }

    def get_random_merge_points(self):

        ridx = np.random.choice(self.to_merge_surf_pts.shape[0], size = self.args.batch_merge_num, replace = True).reshape(-1)

        return {
            'to_merge_surf_pts': self.to_merge_surf_pts[ridx],
            "to_merge_query_pids": self.to_merge_query_pids[ridx],
            "to_merge_surf_normals": self.to_merge_surf_normals[ridx],
            "to_merge_surf_operator": self.to_merge_surf_operator[ridx],
            "to_merge_surf_patch_pair": self.to_merge_surf_patch_pair[ridx],
            'to_merge_ori_pids': self.to_merge_ori_pids[ridx],
            'to_merge_smooth_mask': None, 
            "to_merge_max_merge_count": self.to_merge_max_merge_count,
            "to_merge_max_patch_count": self.to_merge_max_patch_count
        }
