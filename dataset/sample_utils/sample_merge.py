import torch
import trimesh
import numpy as np 
from utils import filter_points_in_redboxes, sample_more_on_merge_box, filter_points_with_pids
from utils import get_box_id_for_pts, extend_filter_points_in_redboxes


## sample surface points equally distributed on the surface
def sample_surface_patch_equal(patch_list, patch_area_list, total_surf_num):

    patch_points = []
    patch_normals = []
    pids = []
    
    patch_area_list = np.array(patch_area_list)
    total_area = np.sum(patch_area_list)

    for chart_id in range(len(patch_list)):
        
        cur_mesh = patch_list[chart_id]
        cur_area = patch_area_list[chart_id]

        ## sample points according to the portion of the patch area
        sample_surf_num = int(total_surf_num * cur_area / total_area)

        ## uniformly sample on this patch
        cur_surface_points, fids = trimesh.sample.sample_surface(cur_mesh, sample_surf_num)
        cur_surface_normals = cur_mesh.face_normals[fids]

        patch_points.append(cur_surface_points)
        patch_normals.append(cur_surface_normals)
        pids.append(np.ones(cur_surface_points.shape[0]) * chart_id)

    patch_points = np.concatenate(patch_points, axis = 0)
    patch_normals = np.concatenate(patch_normals, axis = 0)
    pids = np.concatenate(pids, axis = 0)

    return patch_points, patch_normals, pids
        


def sample_bdr_patch_equal(bdr_info_dict, bdr_distribution_dict, total_surf_num, patch_points, patch_normals, pids):
    
    bdr_portion = int(total_surf_num * 0.5)
    total_bdr_num = sum(bdr_distribution_dict.values())
    
    for bdr_id in bdr_info_dict.keys():

        bdr_points = bdr_info_dict[bdr_id][:, :3]
        bdr_normals = bdr_info_dict[bdr_id][:, 3:]
        bdr_pids = np.ones(bdr_points.shape[0]) * bdr_id

        ## select the number of points according to the portion of the boundary
        bdr_num = int(bdr_portion * bdr_distribution_dict[bdr_id] / total_bdr_num)
        random_idx = np.random.choice(bdr_points.shape[0], bdr_num, replace = True)
        
        bdr_points = bdr_points[random_idx]
        bdr_normals = bdr_normals[random_idx]
        bdr_pids = bdr_pids[random_idx]

        patch_points = np.concatenate([patch_points, bdr_points], axis = 0)
        patch_normals = np.concatenate([patch_normals, bdr_normals], axis = 0)
        pids = np.concatenate([pids, bdr_pids], axis = 0)
    
    return patch_points, patch_normals, pids



def sample_surf_points_for_merge(patch_list, merge_order_list, redboxes, surf_box_num, total_surf_num, patch_area_list, extend_length, open_flag):

    ## uniformly sample on patch surface
    patch_points, patch_normals, pids = sample_surface_patch_equal(patch_list, patch_area_list, total_surf_num)

    patch_points = torch.from_numpy(patch_points).float()
    patch_bids = get_box_id_for_pts(patch_points, redboxes)

    ## initialize the list for merge_surf_points, merge_surf_normals, merge_surf_pids
    merge_surf_points = []
    merge_surf_normals = []
    merge_surf_pids = []
    merge_surf_bids = []

    min_merge_box_num = 2 * surf_box_num

    ## go over merge order
    for merge_piece in merge_order_list:

        ## if no merge happens
        if merge_piece is None or merge_piece['duplicate'] == 0:
            continue

        box_ids = merge_piece['box_id']
        sorted_ids = merge_piece['sorted_ids']

        for box_id in box_ids:
            
            ## filter the points that are in the box_ids                
            filtered_bid_flag = filter_points_in_redboxes(box_id, patch_bids)
            ## count the number of points in the box_ids
            num_points = np.sum(filtered_bid_flag)
            
            ## if the sample points are too few, then sample more for this merge order
            if num_points < min_merge_box_num:
                
                newly_sample_points, newly_sample_normals, newly_sample_pids = sample_more_on_merge_box(sorted_ids, box_id, min_merge_box_num, patch_list, redboxes, extend_length, open_flag)
                
                if not(newly_sample_points is None):
                    merge_surf_points.append(newly_sample_points)
                    merge_surf_normals.append(newly_sample_normals)
                    merge_surf_pids.append(newly_sample_pids)
                    newly_sample_bids = np.ones(newly_sample_points.shape[0]) * box_id
                    merge_surf_bids.append(newly_sample_bids)

            else:
                extend_filtered_bid_flag = extend_filter_points_in_redboxes(patch_points, box_id, redboxes, extend_length)

                selected_points = patch_points[extend_filtered_bid_flag]
                selected_normals = patch_normals[extend_filtered_bid_flag]
                selected_pids = pids[extend_filtered_bid_flag]

                ## select min_merge_box_num points from the filtered points
                random_index = np.random.choice(selected_points.shape[0], min_merge_box_num, replace=False)
                selected_points = selected_points[random_index]
                selected_normals = selected_normals[random_index]
                selected_pids = selected_pids[random_index]

                ## append the selected points, normals, pids to the list
                merge_surf_points.append(selected_points)
                merge_surf_normals.append(selected_normals)
                merge_surf_pids.append(selected_pids)
                selected_bids = np.ones(selected_points.shape[0]) * box_id
                merge_surf_bids.append(selected_bids)

    merge_surf_points = np.concatenate(merge_surf_points, axis = 0)
    merge_surf_normals = np.concatenate(merge_surf_normals, axis = 0)
    merge_surf_pids = np.concatenate(merge_surf_pids, axis = 0)
    merge_surf_bids = np.concatenate(merge_surf_bids, axis = 0)
    
    return merge_surf_points, merge_surf_normals, merge_surf_pids, merge_surf_bids


def presample_points_in_merge_cells(merge_order_list, surface_points, surface_normals, surface_box_ids, surface_pids, impossible_num):

    max_merge_count = 0
    max_patch_count = 0

    merge_surf_points_operator = []
    merge_surf_points_patch_pair = []
    merge_surf_points_pids = []
    merge_surf_points = []
    merge_surf_normals = []
    merge_ori_pids = []

    ## generate merge information 
    for merge_piece in merge_order_list:

        ## if no merge happens
        if merge_piece is None or merge_piece['duplicate'] == 0:
            continue

        box_ids = merge_piece['box_id']
        merge_order = merge_piece['merge_order']
        sorted_ids = merge_piece['sorted_ids']

        ## filter all the related points according to the box ids
        filtered_bid_flag = filter_points_in_redboxes(box_ids, surface_box_ids)

        merge_piece_info = generate_info_base_on_merge(sorted_ids, max_patch_count, merge_order, surface_points, surface_normals, filtered_bid_flag, surface_pids, merge_surf_points, merge_surf_normals, merge_surf_points_pids, merge_surf_points_operator, merge_surf_points_patch_pair, max_merge_count, merge_ori_pids)
        
        merge_surf_points = merge_piece_info[0]
        merge_surf_normals = merge_piece_info[1]
        merge_surf_points_pids = merge_piece_info[2]
        merge_surf_points_operator = merge_piece_info[3]
        merge_surf_points_patch_pair = merge_piece_info[4]
        max_merge_count = merge_piece_info[5]
        max_patch_count = merge_piece_info[6]
        merge_ori_pids = merge_piece_info[7]

    
    merge_surf_points = np.concatenate(merge_surf_points, axis = 0)
    merge_surf_normals = np.concatenate(merge_surf_normals, axis = 0)
    merge_ori_pids = np.concatenate(merge_ori_pids, axis = 0)

    merge_surf_points_operator_real = []
    merge_surf_points_patch_pair_real = []
    merge_surf_points_pids_real = []

    for i in range(len(merge_surf_points_operator)):

        merge_surf_points_operator_piece = np.ones((merge_surf_points_operator[i].shape[0], max_merge_count)) * impossible_num
        merge_surf_points_patch_pair_piece = np.ones((merge_surf_points_patch_pair[i].shape[0], max_merge_count * 2)) * impossible_num
        merge_surf_points_pids_piece = np.ones((merge_surf_points_pids[i].shape[0], max_patch_count)) * impossible_num

        merge_surf_points_operator_piece[:, :merge_surf_points_operator[i].shape[1]] = merge_surf_points_operator[i]
        merge_surf_points_patch_pair_piece[:, :merge_surf_points_patch_pair[i].shape[1]] = merge_surf_points_patch_pair[i]
        merge_surf_points_pids_piece[:, :merge_surf_points_pids[i].shape[1]] = merge_surf_points_pids[i]

        merge_surf_points_operator_real.append(merge_surf_points_operator_piece)
        merge_surf_points_patch_pair_real.append(merge_surf_points_patch_pair_piece)
        merge_surf_points_pids_real.append(merge_surf_points_pids_piece)

    merge_surf_points_operator_real = np.concatenate(merge_surf_points_operator_real, axis = 0)
    merge_surf_points_patch_pair_real = np.concatenate(merge_surf_points_patch_pair_real, axis = 0)
    merge_surf_points_pids_real = np.concatenate(merge_surf_points_pids_real, axis = 0)

    return {'to_merge_surf_operator': torch.from_numpy(merge_surf_points_operator_real), 
            'to_merge_surf_patch_pair': torch.from_numpy(merge_surf_points_patch_pair_real), 
            'to_merge_max_merge_count': max_merge_count,
            'to_merge_max_patch_count': max_patch_count,
            'to_merge_surf_pts': torch.from_numpy(merge_surf_points),
            'to_merge_surf_normals': torch.from_numpy(merge_surf_normals),
            'to_merge_query_pids': torch.from_numpy(merge_surf_points_pids_real),
            'to_merge_ori_pids': torch.from_numpy(merge_ori_pids)}


def generate_info_base_on_merge(new_sorted_ids, max_patch_count, m_order, surface_points, surface_normals, filtered_bid_flag, surface_pids, merge_surf_points, merge_surf_normals, merge_surf_pids, merge_surf_points_operator, merge_surf_points_patch_pair, max_merge_count, merge_ori_pids):
    

    if len(new_sorted_ids) > max_patch_count:
        max_patch_count = len(new_sorted_ids)

    filtered_pid_flag = filter_points_with_pids(new_sorted_ids, surface_pids)
    filtered_surf_flag = np.logical_and(filtered_bid_flag, filtered_pid_flag)

    ## get points & normals for this merge sequence
    surf_points_piece = surface_points[filtered_surf_flag]
    surf_normals_piece = surface_normals[filtered_surf_flag]
    surf_ori_pids_piece = surface_pids[filtered_surf_flag]

    ## get patch ids for the related surf points
    surf_pids_piece = generate_pids_sequence(surf_points_piece, new_sorted_ids)

    merge_surf_points.append(surf_points_piece)
    merge_surf_normals.append(surf_normals_piece)
    merge_surf_pids.append(surf_pids_piece)
    merge_ori_pids.append(surf_ori_pids_piece)

    cur_item_length = surf_points_piece.shape[0]

    surf_operator_piece = []
    surf_patch_pair_piece = []

    merge_count = 0

    for how_to_merge in m_order:

        pid1 = how_to_merge["edge"][0]
        pid2 = how_to_merge["edge"][1]
        operator_sign = how_to_merge["property"]

        ## get operator sign
        surf_operator_one_item = np.ones((cur_item_length, 1)) * operator_sign
        surf_operator_piece.append(surf_operator_one_item)

        ## get patch pair
        surf_patch_pair_one_pair = np.ones((cur_item_length, 2))
        surf_patch_pair_one_pair[:, 0] = pid1
        surf_patch_pair_one_pair[:, 1] = pid2
        surf_patch_pair_piece.append(surf_patch_pair_one_pair)

        merge_count += 1
        
    if merge_count > max_merge_count:
        max_merge_count = merge_count
    
    surf_operator_piece = np.concatenate(surf_operator_piece, axis=1)
    surf_patch_pair_piece = np.concatenate(surf_patch_pair_piece, axis=1)
    merge_surf_points_operator.append(surf_operator_piece)
    merge_surf_points_patch_pair.append(surf_patch_pair_piece)

    return merge_surf_points, merge_surf_normals, merge_surf_pids, merge_surf_points_operator, merge_surf_points_patch_pair, max_merge_count, max_patch_count, merge_ori_pids


def generate_pids_sequence(surf_points_piece, new_sorted_ids):

    surf_pids_piece = np.zeros((surf_points_piece.shape[0],len(new_sorted_ids)))

    for i, pid in enumerate(new_sorted_ids):
        surf_pids_piece[:, i] = np.ones(surf_points_piece.shape[0]) * pid

    return surf_pids_piece
