import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np 
from utils import judge_pts_in_box




def sample_space_pts(adaptive_patch_grids, args, fv_information):

    off_surface_pts_list = []
    off_surface_pid_list = []

    num_patches = len(adaptive_patch_grids)
  
    for chart_id in range(num_patches):

        cur_patch_resolution = 2**adaptive_patch_grids[chart_id]

        bounding_boxes = fv_information[chart_id]
        box_length = 2 / (cur_patch_resolution)

        off_samples = []

        for i in range(bounding_boxes.shape[0]):

            start_p = bounding_boxes[i]
            box_pts = sample_box_points(start_p, box_length, args.box_uniform_num)

            off_samples.append(box_pts)

        off_samples = np.concatenate(off_samples, axis=0)
        off_samples = filter_off_samples(off_samples)
        off_surface_pts_list.append(torch.from_numpy(off_samples).float())
        off_surface_pid_list.append((torch.ones((off_samples.shape[0],1)) * chart_id).float())
    
    off_surface_pts = torch.cat(off_surface_pts_list)
    off_surface_pid = torch.cat(off_surface_pid_list)

    return {
        'spatial_points': off_surface_pts,
        'spatial_pids': off_surface_pid
    }



def filter_off_samples(off_samples,return_flag = False):
    
    start_p = np.array([-1,-1,-1])
    box_length = 2
    filter_flag = judge_pts_in_box(start_p,box_length,off_samples,remain_every_info=True, no_border = True)
    filtered_pts = off_samples[filter_flag]

    if return_flag:
        return filter_flag, filtered_pts
    else:
        return filtered_pts

def sample_box_points(start_point, box_length, box_uniform_num):
    
    uniform_samples = np.random.uniform(start_point,start_point+box_length,(box_uniform_num,3))
    
    return uniform_samples
