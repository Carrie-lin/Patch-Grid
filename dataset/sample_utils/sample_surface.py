import os
import sys
sys.path.append(os.getcwd())
import torch
import trimesh
import numpy as np 

from utils import return_box_cutted_mesh, judge_pts_in_box, sample_points_within_grid
from dataset.sample_utils.sample_jitter import sample_jitter_surface

## get the merge boxes that contains the current patch and are in the current feature volume box 
def get_cur_patch_merge_box(merge_box, chart_id, box_start_point, box_length):
    filtered_box_list = []
    for box in merge_box:
        ## if contains current patch
        if chart_id in box.fids:
            ## if it's in the fv box
            if judge_pts_in_box(box_start_point, box_length, box.center.reshape(-1,3)):
                filtered_box_list.append(box)
    return filtered_box_list

## get the cutted merge patches according to the merge boxes
def get_merge_patch_cutted(cur_patch_cur_fv_merge_boxes, patch_list, cur_patch_id):
    
    cutted_merge_mesh_list = []
    cutted_merge_pid_list = []
    cutted_merge_area_list = []

    for box in cur_patch_cur_fv_merge_boxes:

        ## remove cur_patch_id from the list box.fids
        involve_pid_list = [pid for pid in box.fids if pid != cur_patch_id]
        for pid in involve_pid_list:
            
            merge_patch = patch_list[pid]
            
            box_start_point = box.center - box.half_size
            box_length = 2 * box.half_size

            cutted_mesh = return_box_cutted_mesh(merge_patch, box_start_point, box_length)

            cutted_merge_mesh_list.append(cutted_mesh)
            cutted_merge_pid_list.append(pid)
            cutted_merge_area_list.append(cutted_mesh.area)

    return cutted_merge_mesh_list, cutted_merge_pid_list, cutted_merge_area_list

def sample_surf_normal(mesh, num_points):
    points, fids = trimesh.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[fids]
    return points, normals


## sample surface points equally distributed on the surface
## the points also include merge points
def sample_points_on_surface_equal_box(patch_list, patch_fv_list, patch_resolution, jitter_delta_scale, box_surf_num, bdr_info_dict):

    all_list_points = []
    all_list_normals = []
    all_list_pid = []

    all_list_jitter_points = []
    all_list_jitter_sdfs = []
    all_list_jitter_normals = []
    all_list_jitter_pid = []

    
    ## sample patch points
    for chart_id in range(len(patch_list)):

        cur_surface_points = []
        cur_surface_normals = []
        
        cur_patch_resolution = patch_resolution[chart_id]
        box_length = 2 / 2**cur_patch_resolution 

        cur_fv_info = patch_fv_list[chart_id]
        cur_mesh = patch_list[chart_id]

        ## denser sample for small patches
        sample_surf_num = box_surf_num
        if 2**cur_patch_resolution == 32:
            sample_surf_num = int(1.5 * sample_surf_num)
        if 2**cur_patch_resolution == 64:
            sample_surf_num = 2 * sample_surf_num

        ## go over every feature volume for this patch
        for i in range(cur_fv_info.shape[0]):

            box_start_point = cur_fv_info[i]
            cutted_mesh = return_box_cutted_mesh(cur_mesh, box_start_point, box_length)
            cur_point, cur_normals = sample_surf_normal(cutted_mesh, sample_surf_num)
            cur_surface_points.append(cur_point)
            cur_surface_normals.append(cur_normals)

        cur_surface_points = np.concatenate(cur_surface_points)
        cur_surface_normals = np.concatenate(cur_surface_normals)

        ## sample for surface jitter points
        jitter_data = sample_jitter_surface(cur_surface_points, cur_surface_normals, box_length, jitter_delta_scale)
        all_list_jitter_points.append(jitter_data['points'])
        all_list_jitter_normals.append(jitter_data['normals'])
        all_list_jitter_pid.append(np.ones((jitter_data['points'].shape[0], 1)) * chart_id)
        all_list_jitter_sdfs.append(jitter_data['sdfs'])

        all_list_points.append(cur_surface_points)
        all_list_normals.append(cur_surface_normals)
        all_list_pid.append(np.ones((cur_surface_points.shape[0],1)) * chart_id)
                
    ## sample bdr points for open shape
    for bdr_id in bdr_info_dict.keys():

        cur_bdr_points = []
        cur_bdr_normals = []
         
        bdr_info_piece = bdr_info_dict[bdr_id]
        bdr_points_piece = bdr_info_piece[:,:3]
        bdr_normals_piece = bdr_info_piece[:,3:]

        cur_patch_resolution = patch_resolution[bdr_id]
        box_length = 2 / 2**cur_patch_resolution 

        cur_fv_info = patch_fv_list[bdr_id]

        sample_surf_num = box_surf_num

        if 2**cur_patch_resolution == 32:
            sample_surf_num = int(1.5 * sample_surf_num)

        if 2**cur_patch_resolution == 64:
            sample_surf_num = 2 * sample_surf_num

        ## go over every feature volume for this patch
        for i in range(cur_fv_info.shape[0]):

            box_start_point = cur_fv_info[i]
            ## sample points within the fv grid
            cur_fv_points, cur_fv_normals = sample_points_within_grid(sample_surf_num, box_start_point, box_length, bdr_points_piece, bdr_normals_piece)
            cur_bdr_points.append(cur_fv_points)
            cur_bdr_normals.append(cur_fv_normals)

        cur_bdr_points = np.concatenate(cur_bdr_points)
        cur_bdr_normals = np.concatenate(cur_bdr_normals)
        
        jitter_data = sample_jitter_surface(cur_bdr_points, cur_bdr_normals, box_length, jitter_delta_scale)

        all_list_jitter_points.append(jitter_data['points'])
        all_list_jitter_normals.append(jitter_data['normals'])
        all_list_jitter_pid.append(np.ones((jitter_data['points'].shape[0], 1)) * bdr_id)
        all_list_jitter_sdfs.append(jitter_data['sdfs'])
        
        all_list_points.append(cur_bdr_points)
        all_list_normals.append(cur_bdr_normals)
        all_list_pid.append(np.ones((cur_bdr_points.shape[0],1)) * bdr_id)

    all_list_points = torch.from_numpy(np.concatenate(all_list_points)).float()
    all_list_normals = torch.from_numpy(np.concatenate(all_list_normals)).float()
    all_list_pid = torch.from_numpy(np.concatenate(all_list_pid)).float()

    all_list_jitter_points = torch.from_numpy(np.concatenate(all_list_jitter_points)).float()
    all_list_jitter_normals = torch.from_numpy(np.concatenate(all_list_jitter_normals)).float()
    all_list_jitter_sdfs = torch.from_numpy(np.concatenate(all_list_jitter_sdfs)).float()
    all_list_jitter_pid = torch.from_numpy(np.concatenate(all_list_jitter_pid)).long()

    return {
        'points': all_list_points,
        'normals': all_list_normals,
        'pids': all_list_pid
    }, {
        'jitter_points': all_list_jitter_points,
        'jitter_normals': all_list_jitter_normals,
        'jitter_sdfs': all_list_jitter_sdfs,
        'jitter_pids': all_list_jitter_pid
    }

