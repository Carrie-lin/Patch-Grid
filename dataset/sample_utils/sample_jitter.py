import numpy as np 
import torch
from dataset.sample_utils.sample_space import filter_off_samples


## offset the surface, distribute the points uniformly in the space
def sample_jitter_surface(pts, normals, box_length, jitter_delta_scale):

    cur_jitter_points_out = pts + jitter_delta_scale * box_length * normals
    cur_jitter_points_in = pts - jitter_delta_scale * box_length * normals
    cur_jitter_points = np.concatenate((cur_jitter_points_out, cur_jitter_points_in), axis = 0)

    cur_jitter_normals_out = normals.copy()
    cur_jitter_normals_in = normals.copy()
    cur_jitter_normals = np.concatenate((cur_jitter_normals_out, cur_jitter_normals_in), axis = 0)

    cur_jitter_sdf_out = np.ones((cur_jitter_points_out.shape[0], 1)) * jitter_delta_scale * box_length
    cur_jitter_sdf_in = np.ones((cur_jitter_points_in.shape[0], 1)) * jitter_delta_scale * box_length * (-1)
    cur_jitter_sdfs = np.concatenate((cur_jitter_sdf_out, cur_jitter_sdf_in),axis = 0)
    
    cur_jitter_flag, cur_jitter_points = filter_off_samples(cur_jitter_points, return_flag = True)
    cur_jitter_normals = cur_jitter_normals[cur_jitter_flag]
    cur_jitter_sdfs = cur_jitter_sdfs[cur_jitter_flag]

    return {'points': cur_jitter_points, 
            'normals': cur_jitter_normals, 
            'sdfs': cur_jitter_sdfs
        }

