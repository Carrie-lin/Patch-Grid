import os, sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import trimesh
import yaml
from utils import read_json,get_box_id_for_pts, read_yaml, write_json, draw_colored_points_to_obj
from preparation.lib.mergeutils import RedBox
  


class SmoothType():
    def __init__(self, shape_id, input_dir, save_smooth_root, off_surface_threshold):

        self.save_smooth_root = save_smooth_root
        self.input_dir = input_dir
        self.off_surface_threshold = off_surface_threshold

        ## identify paths
        yaml_path = f'{input_dir}/{shape_id}/{shape_id}.yml'
        redbox_path = f'{save_smooth_root}/{shape_id}/redboxes_adapt.json'
        shape_path = f'{save_smooth_root}/{shape_id}/{shape_id}.obj'
        patch_path = f'{save_smooth_root}/{shape_id}/patches'

        ## collect shape information
        gt_shape = trimesh.load(shape_path, process = False, maintain_order = True, validate = False)
        gt_vertices = gt_shape.vertices
        redboxes = get_redboxes(redbox_path)

        surfaces_points_idx, curve_points_idx = self.read_abc_yaml(yaml_path)
        patch_list = self.read_patch_list(patch_path, len(surfaces_points_idx))

        ## get the map from curve vertices to curve id
        curve_smooth_idx_list, curve_smooth_vertices, curve_special_smooth_idx_list, curve_special_smooth_vertices = self.collect_smooth_curve_list(curve_points_idx, gt_vertices)

        ## if no smooth curves in the shape
        if len(curve_smooth_idx_list) == 0 and len(curve_special_smooth_idx_list) == 0:
            print(f'No smooth curves in shape {shape_id}...')
            self.save_file({},shape_id)
            return 
    
        ## get the map from surface vertices to surface patch
        surface_vertices_idx = self.collect_surface_vertices_list(surfaces_points_idx, patch_list, gt_vertices) ## get the index for surface vertices in each patch
        
        '''
        ## visualize the surface pts
        for i in range(len(surface_vertices_idx)):
            draw_colored_points_to_obj(f'debug/debug_surface_{i}.obj', gt_vertices[surface_vertices_idx[i]])
        exit(0)
        '''

        vertex_belong_list = self.judge_patch_belong(curve_smooth_idx_list, 
                                                     surface_vertices_idx, 
                                                     curve_smooth_vertices, 
                                                     curve_special_smooth_idx_list, 
                                                     curve_special_smooth_vertices, 
                                                     redboxes)
        
        ## assert the length of vertex_belong_list is not 0
        if len(vertex_belong_list) == 0:
            print(f'No smooth curves in shape {shape_id}...')
            self.save_file({},shape_id)
            return 

        ## construct the ideal data structure for smooth connections
        smooth_type_dict = self.generate_smooth_dict(vertex_belong_list)

        ## get and save smooth info json 
        self.save_smooth_info(smooth_type_dict, shape_id)

        ## save the smooth detailed info dict
        self.save_file(smooth_type_dict, shape_id)

    def read_patch_list(self, patch_path, patch_num):

        patch_list = []

        for i in range(patch_num):
            patch_piece_path = f'{patch_path}/{i}.obj'
            patch_piece = trimesh.load(patch_piece_path)
            patch_list.append(patch_piece)

        return patch_list

    def save_file(self,smooth_type_dict,shape_id):
        save_path = f'{self.save_smooth_root}/{shape_id}/smooth_adapt.json'

        ## save json file
        write_json(smooth_type_dict, save_path)
    
    ## save the smoothly connected patch ids 
    def save_smooth_info(self, smooth_type_dict, shape_id):

        save_path = f'{self.save_smooth_root}/{shape_id}/smooth_info.json'
        
        smooth_info = {}

        for key in smooth_type_dict.keys():
            patch_id = key.split(',')[0].replace('[','')
            if patch_id in smooth_info.keys():
                ## add the list in smooth_type_dict[key] to the list smooth_info[patch_id]
                smooth_info[patch_id] += smooth_type_dict[key]
            else:
                smooth_info[patch_id] = []

        for i in smooth_info.keys():
            smooth_info[i] = list(set(smooth_info[i]))
        
        ## save json file 
        write_json(smooth_info, save_path)


    ## get the patch ids connected by smooth curves
    def judge_patch_belong(self, curve_smooth_idx_list, surface_vertices_idx, curve_smooth_vertices, 
                            curve_special_smooth_idx_list, 
                            curve_special_smooth_vertices, 
                            redboxes):
        
        vertex_belong_list = []
        
        ## process for common smooth curves
        if len(curve_smooth_idx_list) > 0:

            ## get the box id for each vertex
            box_id = get_box_id_for_pts(torch.from_numpy(curve_smooth_vertices), redboxes).numpy()
 
            ## we only utilize the points that connect two patches 
            for vertex_id in range(len(curve_smooth_idx_list)):

                vertex_info_piece = {}            
                patch_list = self.select_patch_belong_for_one_p(curve_smooth_idx_list[vertex_id], surface_vertices_idx)
                vertex_info_piece['box'] = int(box_id[vertex_id])
                vertex_info_piece['patch'] = patch_list

                if len(patch_list) == 2:
                    vertex_belong_list.append(vertex_info_piece)
        
        ## special process for the smooth curves with only two points
        if len(curve_special_smooth_idx_list) > 0:  

            special_box_id = get_box_id_for_pts(torch.from_numpy(curve_special_smooth_vertices), redboxes).numpy()

            for vertex_id in range(0,len(curve_special_smooth_idx_list),2):

                vertex_info_piece_1 = {}     
                vertex_info_piece_2 = {}

                ## get two patch lists of the two points       
                patch_list_1 = self.select_patch_belong_for_one_p(curve_special_smooth_idx_list[vertex_id], surface_vertices_idx)
                patch_list_2 = self.select_patch_belong_for_one_p(curve_special_smooth_idx_list[vertex_id+1], surface_vertices_idx)

                ## find overlap of patch_list_1 and patch_list_2
                patch_list = []
                for patch_id in patch_list_1:
                    if patch_id in patch_list_2:
                        patch_list.append(patch_id)
                
                vertex_info_piece_1['box'] = int(special_box_id[vertex_id])
                vertex_info_piece_1['patch'] = patch_list

                vertex_info_piece_2['box'] = int(special_box_id[vertex_id+1])
                vertex_info_piece_2['patch'] = patch_list

                vertex_belong_list.append(vertex_info_piece_1)
                vertex_belong_list.append(vertex_info_piece_2)

        return vertex_belong_list
    
    ## construct the ideal data structure for smooth connections
    ## ideal form: key:[patch_id,box_id] = [patch_a,patch_b,patch_c,...,patch_n]
    def generate_smooth_dict(self, vertex_belong_list):

        smooth_dict = {}

        for vertex_info in vertex_belong_list:

            assert len(vertex_info['patch']) == 2

            key_a = str([vertex_info['patch'][0],vertex_info['box']])
            key_b = str([vertex_info['patch'][1],vertex_info['box']])

            patch_for_a = vertex_info['patch'][1]
            patch_for_b = vertex_info['patch'][0]

            if not (key_a in smooth_dict.keys()):
                smooth_dict[key_a] = [patch_for_a]
            if not (key_b in smooth_dict.keys()):
                smooth_dict[key_b] = [patch_for_b]
            if not(patch_for_a in smooth_dict[key_a]):
                smooth_dict[key_a].append(patch_for_a)
            if not(patch_for_b in smooth_dict[key_b]):
                smooth_dict[key_b].append(patch_for_b)

        return smooth_dict 

    ## go over the vertex set for each patch and find the patch indices that contain the vertex
    def select_patch_belong_for_one_p(self, vertex_idx, surface_vertices_idx):
        patch_list = []
        for patch_id in range(len(surface_vertices_idx)):
            surface_piece_list = surface_vertices_idx[patch_id]
            if vertex_idx in surface_piece_list:
                patch_list.append(patch_id)
        return patch_list

    ## collect the real surface vertices for each patch
    ## filter those outliers in abc dataset
    def collect_surface_vertices_list(self, surfaces_points_idx, patch_list, gt_vertices):

        surface_vertices_idx = []
        surface_length = len(surfaces_points_idx)

        for surface_id in range(surface_length):
            surface_vertices_idx.append(surfaces_points_idx[surface_id]['vert_indices'])
        
        surface_idx_remain_list = []

        ## delete those outliers by double check
        ## compute the sdf value of the points at the surface 
        ## if it's larger than the threshold, delete the corresponding point
        for surface_id in range(surface_length):
            gt_patch = patch_list[surface_id]
            surface_idx_piece = surfaces_points_idx[surface_id]['vert_indices']
            #print('total gt ver length:',gt_vertices.shape)
            #print('max idx:',min(surface_idx_piece))
            #continue
            surface_idx_piece = np.array(surface_idx_piece)
            ## find the index that is larger than the total length of gt_vertices and remove those index
            surface_idx_piece = surface_idx_piece[surface_idx_piece < gt_vertices.shape[0]]

            surface_pts_piece = gt_vertices[surface_idx_piece]
            sdf = abs(trimesh.proximity.signed_distance(gt_patch, surface_pts_piece))
            right_pts_flag = sdf < self.off_surface_threshold
            #print('filter num:', len(right_pts_flag) - right_pts_flag.sum())
            surface_idx_piece = np.array(surface_idx_piece)
            filtered_surface_idx_piece = surface_idx_piece[right_pts_flag]
            filtered_surface_idx_piece = filtered_surface_idx_piece.tolist()
            surface_idx_remain_list.append(filtered_surface_idx_piece)

        return surface_idx_remain_list


    ## collect the points in the smooth curves
    def collect_smooth_curve_list(self,curve_points,gt_vertices):

        ## collect the points in smooth curves 
        ## common smooth curves
        curve_smooth_idx_list = []
        curve_real_vertices = None
        ## special smooth curves (curve with only two points)
        curve_special_smooth_idx_list = []
        curve_special_vertices = None

        cid = 0 ## curve id

        for curve_piece in curve_points:
            sharp_type = (curve_piece['sharp'])
            ## if it's smooth connection
            if sharp_type == False:
                ## remove the first and the last point for common smooth curves (or else will introduce sharp points)
                for idx in range(1, len(curve_piece['vert_indices'])-1):
                    curve_smooth_idx_list.append(curve_piece['vert_indices'][idx]) ## the index of the vertex in the whole shape
                ## if this curve only contains two points, mark it as special smooth curve
                if len(curve_piece['vert_indices']) == 2:
                    curve_special_smooth_idx_list.append(curve_piece['vert_indices'][0])
                    curve_special_smooth_idx_list.append(curve_piece['vert_indices'][1])
            cid += 1
        
        ## if the shape has smooth curves
        if len(curve_smooth_idx_list) > 0:
            curve_smooth_idx = np.array(curve_smooth_idx_list)
            curve_real_vertices = gt_vertices[curve_smooth_idx] ## get the real coordinates of the smooth curve vertices

        ## if the shape has special smooth curves (curve with only two points)
        if len(curve_special_smooth_idx_list) > 0:
            curve_special_smooth_idx = np.array(curve_special_smooth_idx_list)
            curve_special_vertices = gt_vertices[curve_special_smooth_idx] ## get the real coordinates of the smooth curve vertices

        return curve_smooth_idx_list, curve_real_vertices, curve_special_smooth_idx_list, curve_special_vertices

    def read_abc_yaml(self,yaml_path):

        file = open(yaml_path, 'r', encoding="utf-8")
        file_data = file.read()
        file.close()
        data = yaml.load(file_data,Loader=yaml.FullLoader)
        
        return data['surfaces'], data['curves']

## load the box file and build the corresponding box list
def get_redboxes(redbox_path):

    file_redboxes = redbox_path
    list_redboxes = []
    if os.path.exists(file_redboxes):
        redboxes = read_json(file_redboxes)
        for rbx in redboxes:
            half_size = rbx['half_size'] 
            rbxx = RedBox(np.array(rbx['center']), half_size)
            rbxx.set_face_ids(rbx['face_ids'])
            rbxx.set_height(rbx['height'])
            list_redboxes.append(rbxx)

    return list_redboxes


## process the smooth type data for direct use 
def process_smooth_type(smooth_type):
    ready_to_use_smooth_type = {}
    for patch_box_key in smooth_type.keys():
        patch_key = (eval(patch_box_key))[0]
        if not (patch_key in ready_to_use_smooth_type.keys()):
            ready_to_use_smooth_type[patch_key] = {}
        box_key = (eval(patch_box_key))[1]
        for smooth_patch in smooth_type[patch_box_key]:
            if not (smooth_patch in ready_to_use_smooth_type[patch_key].keys()):
                ready_to_use_smooth_type[patch_key][smooth_patch] = []
            ready_to_use_smooth_type[patch_key][smooth_patch].append(box_key)
    return ready_to_use_smooth_type


## process self design smooth connections
def compute_smooth_from_design(shape_id, save_smooth_root):

    ## load box and smooth connection information
    smooth_dict = read_json(f'{save_smooth_root}/{shape_id}/smooth_info.json')
    smooth_dict = {int(k):v for k,v in smooth_dict.items()}
    merge_order = read_json(f'{save_smooth_root}/{shape_id}/merge_order_adapt.json')

    ## add box information to smooth connections
    new_smooth_dict = {}

    for merge_piece in merge_order:
        if not (merge_piece is None):
            box_list = merge_piece['box_id']
            patch_list = merge_piece['patch_ids']
            for box_piece in box_list:
                for patch_piece in patch_list:
                    smooth_piece = {}
                    add_flag = False
                    smooth_piece_key = str([patch_piece,box_piece])
                    smooth_piece[smooth_piece_key] = []
                    if patch_piece in smooth_dict.keys():
                        neigh_list = smooth_dict[patch_piece]
                        for other_patch in patch_list:
                            if not (other_patch == patch_piece):
                                if other_patch in neigh_list:
                                    smooth_piece[smooth_piece_key].append(other_patch)
                                    add_flag = True   
                    if add_flag:
                        new_smooth_dict[smooth_piece_key] = smooth_piece[smooth_piece_key]
    out_save_file(new_smooth_dict, save_smooth_root, shape_id)
    return new_smooth_dict
      
def out_save_file(smooth_type_dict,save_smooth_root,shape_id):
    
    save_path = f'{save_smooth_root}/{shape_id}/smooth_adapt.json'
    write_json(smooth_type_dict, save_path)                   
                      
