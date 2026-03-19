import os
import sys
sys.path.append(os.getcwd())

import torch
from utils import filter_points_with_pids_torch
from loss.merge_utils import query_fv_id, bin_count_fvid

def compute_patch_normal_loss(pred_norm, gt_norm):

    normals_loss = (pred_norm - gt_norm).norm(2, dim=1)
    normal_loss_value = normals_loss.mean()

    return normals_loss, normal_loss_value

def compute_patch_spatial_loss(spatial_pred):

    off_surf_penalty = torch.exp(-1e2 * torch.abs(spatial_pred)).mean()

    return off_surf_penalty


def detach_static_sdf(pred_sdfs_merge_list, to_merge_query_pids, update_pid_list):
    for col in range(to_merge_query_pids.shape[1]):
        cur_sdf = pred_sdfs_merge_list[:,col]
        cur_query_pids = to_merge_query_pids[:,col].reshape(-1)
        cur_update_mask = filter_points_with_pids_torch(update_pid_list, cur_query_pids).reshape(-1)
        cur_detach_mask = ~cur_update_mask
        pred_sdfs_merge_list[cur_detach_mask, col] = cur_sdf[cur_detach_mask].detach()
    return pred_sdfs_merge_list

## normalize the merge loss according to the feature volumes
def normalize_merge_loss(to_merge_surf_pts, out_sdf, out_pid, patch_res_arr, patch_grid_dict_map):
    
    ## get the feature volume id for each merge loss
    fv_id = query_fv_id(patch_grid_dict_map, patch_res_arr, to_merge_surf_pts, out_pid)
    assert torch.min(fv_id) > -1

    ## compute mean in each feature volume
    fv_mean = bin_count_fvid(fv_id, out_sdf)

    ## compute final mean value of the merge loss
    fv_mean = fv_mean.mean()

    return fv_mean


## filter the merge boxes that contain smooth joint
def filter_smooth_merge_box(to_merge_surf_pts, out_sdf, out_pid, to_merge_smooth_mask):

    to_merge_surf_pts_smooth = to_merge_surf_pts[to_merge_smooth_mask]
    out_sdf_smooth = out_sdf[to_merge_smooth_mask]
    out_pid_smooth = out_pid[to_merge_smooth_mask]

    return to_merge_surf_pts_smooth, out_sdf_smooth, out_pid_smooth

def compute_merge_loss(args, pred_sdfs_merge_list, to_merge_surf_operator, to_merge_surf_patch_pair, \
                    to_merge_max_merge_count, to_merge_max_patch_count, impossible_num, to_merge_ori_pids, \
                    to_merge_query_pids, to_merge_surf_pts, patch_res_arr, patch_grid_dict_map, epoch, \
                    update_pid_list, to_merge_smooth_mask):

    ## detach the pred sdf that belongs to other patches
    if not update_pid_list is None:
        pred_sdfs_merge_list = detach_static_sdf(pred_sdfs_merge_list, to_merge_query_pids, update_pid_list)

    out_sdf = torch.zeros((pred_sdfs_merge_list.shape[0], 1), dtype = torch.float32).cuda()
    #out_normal = torch.zeros((pred_normal_merge_list.shape[0], 3), dtype = torch.float32).cuda()
    out_pid = torch.zeros((pred_sdfs_merge_list.shape[0], 1), dtype = torch.long).cuda()

    ## get the abs of to_merge_surf_operator and replace all the abs(impossible_num) to be zero
    operator_count = torch.abs(to_merge_surf_operator)
    operator_count[to_merge_surf_operator == impossible_num] = 0
    operator_count = operator_count.sum(dim = 1)

    for merge_id in range(to_merge_max_merge_count):
        
        ## get the valid merge operations
        merge_mask = (to_merge_surf_operator[:,merge_id] != impossible_num)

        ## get the valid operator
        operator = to_merge_surf_operator[merge_mask, merge_id].reshape(-1)

        ## get the operator count for current points
        current_operator_count = operator_count[merge_mask].reshape(-1)

        ## get the rows and columes indices for pred_sdfs_merge_list
        pid1 = to_merge_surf_patch_pair[merge_mask, merge_id * 2].reshape(-1) 
        pid2 = to_merge_surf_patch_pair[merge_mask, merge_id * 2 + 1].reshape(-1) 

        pred_sdf_pid1 = pred_sdfs_merge_list[merge_mask, pid1]
        pred_sdf_pid2 = pred_sdfs_merge_list[merge_mask, pid2]

        ## if the operator is -1, then get the smaller one of pred_sdf_pid1 and pred_sdf_pid2, and return the pid for the smaller one (if choose pred_sdf_pid1, then return pid1, else, return pid2)
        ## if the operator is 1, then get the larger one of pred_sdf_pid1 and pred_sdf_pid2, and return the pid for the larger one (if choose pred_sdf_pid1, then return pid1, else, return pid2)
        
        # Get the mask for smaller and larger elements
        smaller_mask = (pred_sdf_pid1 < pred_sdf_pid2).reshape(-1)
        larger_mask = (pred_sdf_pid1 > pred_sdf_pid2).reshape(-1)

        # select the pid based on the operator and the mask
        selected_pid = torch.where(operator == -1, pid1 * smaller_mask + pid2 * larger_mask, pid1 * larger_mask + pid2 * smaller_mask)
    
        # get the real pid according to the selected pid
        selected_real_pid = to_merge_query_pids[merge_mask, selected_pid]

        # select the value based on the operator and the mask
        selected_sdfs = torch.where(operator == -1, pred_sdf_pid1 * smaller_mask + pred_sdf_pid2 * larger_mask, pred_sdf_pid1 * larger_mask + pred_sdf_pid2 * smaller_mask)

        ## put the selected sdf and normal to the pid1 colume of pred_sdfs_merge_list and pred_normal_merge_list
        pred_sdfs_merge_list[merge_mask, pid1] = selected_sdfs

        ## get the points that end at current merge
        current_batch_end_mask = (current_operator_count == (merge_id + 1)).reshape(-1)
        end_mask = (operator_count == (merge_id + 1)).reshape(-1)

        ## assign the merged value to the end points
        out_sdf[end_mask] = (selected_sdfs[current_batch_end_mask]).reshape(-1,1)
        out_pid[end_mask] = (selected_real_pid[current_batch_end_mask]).reshape(-1,1)

    out_sdf = out_sdf.abs()

    ## constrain out_sdf to be zero
    ## normalize the merge loss according to the feature volumes
    sdf_loss = normalize_merge_loss(to_merge_surf_pts, out_sdf, out_pid, patch_res_arr, patch_grid_dict_map)
    
    merge_loss = args.merge_surf * sdf_loss 

    return {"merge_loss": merge_loss}
      