import torch 

## construct the map array that could points to fv count number
def map_patch_grid_dict(patch_grid_dict, max_code_res, num_patches):

    patch_grid_dict_map = (-1) * torch.ones(num_patches, max_code_res, max_code_res, max_code_res).long()
    
    fv_count = 0
    for pid in patch_grid_dict.keys():
        fv_list = patch_grid_dict[pid]
        for fv in fv_list:
            patch_grid_dict_map[pid, fv[0], fv[1], fv[2]] = fv_count
            fv_count += 1

    return patch_grid_dict_map


def query_fv_id(patch_grid_dict_map, patch_res_arr, x, pid):

    ## x is a float array from -1 to 1
    x = (x + 1) / 2

    special_res = patch_res_arr[pid].reshape(-1,1)
    x = x * (special_res)
    xlong = x.long() 

    return pts_index(patch_grid_dict_map, xlong, pid)

def pts_index(patch_grid_dict_map, xlong, pid):

    device = xlong.device
    xlong = xlong.detach().cpu().numpy()
    fv_id = patch_grid_dict_map[pid.reshape(-1), xlong[:, 0], xlong[:, 1], xlong[:, 2]].to(device).long()

    return fv_id

def bin_count_fvid(fv_id, sdf_value):

    # Prepare tensors
    max_id = torch.max(fv_id).item()
    sums = torch.zeros(max_id + 1, dtype=torch.float).to(fv_id.device)
    counts = torch.zeros(max_id + 1, dtype=torch.float).to(fv_id.device)

    # Compute sum for each id
    sums.scatter_add_(0, fv_id, sdf_value.squeeze())

    # Compute count for each id
    counts.scatter_add_(0, fv_id, torch.ones_like(fv_id, dtype=torch.float))

    ## remove those data with counts == 0
    mask = counts > 0
    counts = counts[mask]
    sums = sums[mask]

    # Compute mean for each id
    means = sums / counts
    return means
