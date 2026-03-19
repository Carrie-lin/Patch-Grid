import matplotlib
import os
import time
import numpy as np
import torch
from matplotlib import cm
import json
from collections import OrderedDict
from glob import glob
import trimesh
import math
from scipy.spatial import cKDTree
import yaml
from tqdm import tqdm
from metrics.lib.torchgp import compute_sdf


def compute_pts_sdf(mesh, pts, batch_size):

    pts = torch.from_numpy(pts).float()

    v = mesh.vertices
    v = torch.from_numpy(v).cuda().float()
    f = mesh.faces
    f = torch.from_numpy(f).cuda().long()

    iter_num = math.ceil(pts.shape[0]/batch_size)

    for i in range(iter_num):

        cur_batch_process_num = min(batch_size, pts.shape[0] - i * batch_size)
        start_index = i * batch_size

        batch_pts = pts[start_index:cur_batch_process_num + start_index, :].cuda()       

        if i == 0:

            d  = compute_sdf(v, f, batch_pts)
            d = d[...,None]
            d = d.cpu().reshape(-1)

        else:

            batch_d = compute_sdf(v, f, batch_pts)
            batch_d = batch_d[...,None]
            batch_d = batch_d.cpu().reshape(-1)
            d = np.concatenate((d, batch_d), axis=0)
            
    return d


def subtle_process_for_pts(global_q):
    detect_boundary_flag = (global_q >= 1)
    global_q[detect_boundary_flag] = 0.999
    return global_q


def sample_pts_in_merge_grids(merge_grids, sample_resolution, scale_facor=1):
    
    pts_list = []

    for bid, rbx in tqdm(enumerate(merge_grids)):

        assert sample_resolution >= 2**rbx.height
        N = int(sample_resolution / (2 ** (rbx.height))) + 1
        rbx.build_box(N, custom_half_size = rbx.half_size * scale_facor) 

        grid_pts = rbx.get_grid_pts() 
        global_q = torch.FloatTensor(grid_pts)
        global_q = subtle_process_for_pts(global_q)
        global_q = global_q
        pts_list.append(global_q)

    return pts_list


def adjust_grid_points_with_append_layer(append_layer_num, rbx, N):
    if append_layer_num > 0:
        ori_step_len = (2 * rbx.half_size) / (N - 1)
        N = 2 * append_layer_num + N
        adjust_half_size = append_layer_num * ori_step_len + rbx.half_size
    else:
        adjust_half_size = rbx.half_size 
        N = N
    return N, adjust_half_size


def filter_off_samples(off_samples,return_flag = False):
    
    start_p = np.array([-1,-1,-1])
    box_length = 2
    filter_flag = judge_pts_in_box(start_p,box_length,off_samples,remain_every_info=True, no_border = True)
    filtered_pts = off_samples[filter_flag]
    if return_flag:
        return filter_flag, filtered_pts
    else:
        return filtered_pts
    
def read_yaml(yaml_path):
    file = open(yaml_path, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    data = yaml.load(file_data,Loader=yaml.FullLoader)
    return data

## load the resolution and feature volume of each patch
def load_patch_reso_fv(data_prefix, shape_name):
    patch_resolution_dict_path = f'{data_prefix}/{shape_name}/adaptive_resolution.json'
    patch_resolution_dict = read_json(patch_resolution_dict_path)
    patch_fv_dict = {}
    for key in patch_resolution_dict:
        patch_fv_path = f'{data_prefix}/{shape_name}/bounds/box_p{key}_r{int(2**patch_resolution_dict[key])}_surface.bin'
        patch_fv_dict[key] = (np.fromfile(patch_fv_path)).reshape(-1,3) # float64
    return patch_resolution_dict, patch_fv_dict

def write_octree_obj_file(filename, V, F=None, C=None, vid_start=1):
    with open(filename, 'w') as f:
        if C is not None:
            for Vi, Ci in zip(V, C):
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]} {Ci[0]} {Ci[1]} {Ci[2]}\n")
        else:
            for Vi in V:
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]}\n")
        if F is not None:
            for Fi in F:
                if len(Fi)==3:
                    f.write(f"f {Fi[0]+vid_start} {Fi[1]+vid_start} {Fi[2]+vid_start}\n")
                else:
                    f.write(f"f {Fi[0]+vid_start} {Fi[1]+vid_start} {Fi[2]+vid_start} {Fi[3]+vid_start}\n")
    

## judge if the box has intersected with the mesh
def judge_if_intersect(mesh,box_start_point,box_length):
    cutted_mesh = return_box_cutted_mesh(mesh,box_start_point,box_length)
    if cutted_mesh is None:
        return False  
    else:
        return True

## return the cutted mesh
def return_box_cutted_mesh(mesh,box_start_point,box_length):

    box_end_point = box_start_point + box_length
    plane_origin_list = [box_start_point,box_start_point,box_start_point,box_end_point,box_end_point,box_end_point]
    plane_normal_list = [[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]]
    cutted_mesh = trimesh.intersections.slice_mesh_plane(mesh,plane_normal_list[0],plane_origin_list[0])
    if (cutted_mesh.vertices).shape[0]>0:
        for j in range(1,6):
            cutted_mesh = trimesh.intersections.slice_mesh_plane(cutted_mesh,plane_normal_list[j],plane_origin_list[j])
            #print(cutted_mesh.vertices)
            #cutted_mesh.export(f'mesh{i}.obj')
            if (cutted_mesh.vertices).shape[0] == 0:
                return None
        return cutted_mesh
    else:
        return None

## subdivide the box of ori_length into small grids of new_length
def subdivide_box(start_p, ori_length, new_length):

    num_grids = int(ori_length / new_length)
    grids = np.mgrid[0:num_grids, 0:num_grids, 0:num_grids]
    points = start_p.reshape(-1, 1, 1, 1) + grids * new_length
    points = points.reshape(3, -1).T

    return points


## judge if the point is in the box list
def is_point_in_boxes(point, boxes, cube_length):
    
    point_expanded = np.tile(point, (len(boxes), 1))
    in_boxes = np.logical_and(boxes <= point_expanded, point_expanded < boxes + cube_length)
    in_boxes_all_dims = np.all(in_boxes, axis=1)

    return in_boxes_all_dims

## judge if each item in small set is contained in big set
def is_subset(small_set, big_set):

    if small_set.shape[0] > big_set.shape[0]:
        return False 

    # Create a structured array for small_set and big_set that allows us to compare rows
    dtype = [('f{}'.format(i), small_set.dtype) for i in range(small_set.shape[1])]
    small_set_structured = np.core.records.fromarrays(small_set.transpose(), dtype=dtype)
    big_set_structured = np.core.records.fromarrays(big_set.transpose(), dtype=dtype)

    # Check if each row of small_set is in big_set
    subset = np.isin(small_set_structured, big_set_structured)

    # If all elements of subset are True, then every row in small_set is in big_set
    return subset.all()

def filter_points_in_redboxes(box_ids, surface_box_ids):

    #print(surface_box_ids.shape)
    box_ids = np.array(box_ids)
    
    ## filter the surface_box_ids according to the box_ids
    filter_flag = np.isin(surface_box_ids, box_ids).reshape(-1)

    return filter_flag

def extend_filter_points_in_redboxes(patch_points, box_id, redboxes, extend_box_length):

    ## redboxes is a list, box_ids is the list for the target index for items in redboxes
    ## get the corresponding box information according to box_ids
    redbox_start_points = []
    redbox_lens = []

    redbox = redboxes[box_id]
    redbox_start_points.append(redbox.center - redbox.half_size)
    redbox_lens.append(2 * redbox.half_size)

    redbox_start_points = torch.Tensor(redbox_start_points)
    redbox_lens = torch.Tensor(redbox_lens).reshape(-1, 1)

    filtered_points, filter_mask = filter_points_in_boxes(redbox_start_points, redbox_lens, patch_points, extend_box_length)

    return filter_mask

def sample_more_on_merge_box(sorted_ids, box_id, min_merge_box_num, patch_list, redboxes, extend_length, open_flag = False):

    cur_patch_list = []
    if open_flag:
        cur_patch_list.append(patch_list[0])
    else:
        for pid in sorted_ids:
            cur_patch_list.append(patch_list[pid])
        
    sample_points_list = []
    sample_normals_list = []
    sample_pids_list = []

    box_start_points = redboxes[box_id].center - redboxes[box_id].half_size
    box_length = 2 * redboxes[box_id].half_size

    box_start_points = box_start_points - extend_length
    box_length = box_length + 2 * extend_length

    area_list = []
    cutted_patch_list = []

    for patch in cur_patch_list:
        cutted_patch = return_box_cutted_mesh(patch, box_start_points, box_length)
        area_list.append(cutted_patch.area)
        cutted_patch_list.append(cutted_patch)

    area_list = np.array(area_list)

    ## sample points on patches according to their area 
    total_sample_num = min_merge_box_num
    sample_num_list = (area_list / area_list.sum() * total_sample_num).astype(int)


    for i in range(len(sample_num_list)):
        cur_pid = sorted_ids[i]
        sample_points, sample_normals = sample_points_on_mesh(cutted_patch_list[i], sample_num_list[i])
        sample_pids = np.ones(sample_points.shape[0]) * cur_pid

        sample_points_list.append(sample_points)
        sample_normals_list.append(sample_normals)
        sample_pids_list.append(sample_pids)

    if len(sample_points_list) == 0:
        return None, None, None
    else:
        newly_sample_points = np.concatenate(sample_points_list)
        newly_sample_normals = np.concatenate(sample_normals_list)
        newly_sample_pids = np.concatenate(sample_pids_list)
                
        return newly_sample_points, newly_sample_normals, newly_sample_pids

def sample_more_on_merge_box_open(sorted_ids, box_id, min_merge_box_num, patch_list, redboxes, extend_length, bdr_distribution_dict, bdr_info_dict):

    cur_patch_list = []
    for pid in sorted_ids:
        if pid == 0:
            cur_patch_list.append(patch_list[pid])
        else:
            cur_patch_list.append(bdr_info_dict[pid])

    sample_points_list = []
    sample_normals_list = []
    sample_pids_list = []

    box_start_points = redboxes[box_id].center - redboxes[box_id].half_size
    box_length = 2 * redboxes[box_id].half_size

    ## extend the box 
    box_start_points = box_start_points - extend_length
    box_length = box_length + 2 * extend_length

    area_list = []
    cutted_patch_list = []

    for i in range(len(cur_patch_list)):

        real_pid = sorted_ids[i]
        
        sample_info = cur_patch_list[i]
        area_list.append(1)
        if real_pid == 0:
            cutted_patch = return_box_cutted_mesh(sample_info, box_start_points, box_length)
            cutted_patch_list.append(cutted_patch)
            
        else:
            bdr_points = sample_info[:,:3]
            filter_flag = judge_pts_in_box(box_start_points, box_length, bdr_points, remain_every_info = True)
            cutted_bdr_info = sample_info[filter_flag]
            cutted_patch_list.append(cutted_bdr_info)

    ## sample equally on boundary and surface
    total_sample_num = min_merge_box_num
    sample_num_list = (area_list / area_list.sum() * total_sample_num).astype(int)

    for i in range(len(sample_num_list)):

        cur_pid = sorted_ids[i]
        
        ## if it's on the patch
        if cur_pid == 0:
            sample_points, sample_normals = sample_points_on_mesh(cutted_patch_list[i], sample_num_list[i])
        ## if it's on the boundary
        else:
            sample_points = cutted_patch_list[i][:,:3]
            sample_normals = cutted_patch_list[i][:,3:6]

        sample_pids = np.ones(sample_points.shape[0]) * cur_pid

        sample_points_list.append(sample_points)
        sample_normals_list.append(sample_normals)
        sample_pids_list.append(sample_pids)

    if len(sample_points_list) == 0:
        return None, None, None
    else:
        newly_sample_points = np.concatenate(sample_points_list)
        newly_sample_normals = np.concatenate(sample_normals_list)
        newly_sample_pids = np.concatenate(sample_pids_list)
                
        return newly_sample_points, newly_sample_normals, newly_sample_pids


def sample_points_on_mesh(cur_mesh, sample_surf_num):
    cur_surface_points, fids = trimesh.sample.sample_surface(cur_mesh, sample_surf_num)
    cur_surface_normals = cur_mesh.face_normals[fids]
    return cur_surface_points, cur_surface_normals

def filter_query_pids_with_update_pids(query_pids, update_pids):
    ## go over every colume of query_pids of shape(N,M)
    mask_list = []
    for col in range(query_pids.shape[1]):
        cur_query_pids = query_pids[:,col].reshape(-1)
        cur_mask = filter_points_with_pids(update_pids, cur_query_pids).reshape(-1,1)
        mask_list.append(cur_mask)
    ori_mask = np.concatenate(mask_list, axis = 1)
    final_mask = np.any(ori_mask, axis = 1).reshape(-1)
    return final_mask

def filter_points_with_pids(target_pids, surface_points_pids):

    target_pids = np.array(target_pids)
    
    ## filter the surface_box_ids according to the box_ids
    filter_flag = np.isin(surface_points_pids, target_pids).reshape(-1)

    return filter_flag

def filter_points_with_pids_torch(target_pids, surface_points_pids):

    ## filter the surface_box_ids according to the box_ids
    filter_flag = torch.zeros_like(surface_points_pids, dtype=torch.bool)
    for i in range(len(target_pids)):
        filter_flag |= (surface_points_pids == target_pids[i])
    
    filter_flag = filter_flag.view(-1)
    
    return filter_flag

def get_bdr_num(data_prefix, shape_name):

    bdr_files = glob(f'{data_prefix}/{shape_name}/bdr*.npy')
    bdr_num = len(bdr_files)
    
    return bdr_num


## judge whether the points are in the box (return overall or per point flag)
def judge_pts_in_box(start_p, box_length, pts_arr,remain_every_info=False,no_border = False, border_value=0.00001):

    if no_border:
        thres = border_value
        x_flag = np.bitwise_and(pts_arr[:,0] > start_p[0] + thres, pts_arr[:,0] < (start_p[0] + box_length - thres))
        y_flag = np.bitwise_and(pts_arr[:,1] > start_p[1] + thres, pts_arr[:,1] < (start_p[1] + box_length - thres))
        z_flag = np.bitwise_and(pts_arr[:,2] > start_p[2] + thres, pts_arr[:,2] < (start_p[2] + box_length - thres))
    else:
        x_flag = np.bitwise_and(pts_arr[:,0] > start_p[0], pts_arr[:,0] < (start_p[0] + box_length))
        y_flag = np.bitwise_and(pts_arr[:,1] > start_p[1], pts_arr[:,1] < (start_p[1] + box_length))
        z_flag = np.bitwise_and(pts_arr[:,2] > start_p[2], pts_arr[:,2] < (start_p[2] + box_length))
    in_box_flag = np.bitwise_and(x_flag,y_flag)
    in_box_flag = np.bitwise_and(in_box_flag,z_flag)
    if remain_every_info:
        return in_box_flag
    else:
        return (in_box_flag.sum() > 0)

## output the meshes and the id for the patches
def get_patches(dataset_path):

    patch_num = get_num_patches(dataset_path)
    patch_list = []

    for pid in range(patch_num):
        fname = f'{dataset_path}/patches/{pid}.obj'
        patch = trimesh.load(fname)
        patch_list.append(patch)

    return patch_list

## get the number of patches 
def get_num_patches(dataset_path, connect_info = None):
    ## to get the number of patches from data folder (for common abc shapes)
    if connect_info is None:
        fnames = glob(os.path.join(dataset_path,'patches','*.obj'))
        expname = dataset_path.split('/')[-1]
        for filename in fnames:
            if os.path.basename(filename) == expname+'.obj':
                fnames.remove(filename)
        N_patches = len(fnames)
    ## to get the number of patches from connect_info (for open boundaries & edited shapes)
    else:
        try:
            N_patches = connect_info.shape[0]
        except:
            N_patches = 1

    return N_patches

# the bridge between an argparse based approach and a non-argparse one
def setparam(args, param, paramstr):
    argsparam = getattr(args, paramstr, None)
    if param is not None or argsparam is None:
        return param
    else:
        return argsparam

# create the directory if it does not exist
def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# get the box id for each point (boxes are organized in octree)
def get_box_id_for_pts(surf_pts, redboxes):

    half_size_list,height_list,bid_list,center_list = collect_grid_info(redboxes)

    kdtree_layer_dict = build_each_layer_kdtree(half_size_list,height_list,bid_list,center_list) ## from 4 to N --> go deeper 
    box_belong = torch.ones((surf_pts.shape[0],1)) * (-100)
    box_belong = box_belong.int()
    for height in kdtree_layer_dict.keys():

        cur_height_tree = kdtree_layer_dict[height]['kdtree']
        cur_height_center_list = kdtree_layer_dict[height]['center']
        cur_height_half_size_list = kdtree_layer_dict[height]['half_size']
        cur_height_bid_list = kdtree_layer_dict[height]['bid']
        _, inside_mask = cur_height_tree.query(surf_pts, k=1, p=1, workers=4)

        if cur_height_bid_list.shape[0] == 0:
            continue
        real_bid_list = torch.from_numpy(cur_height_bid_list[inside_mask]).reshape(-1,1)
        real_bid_list = real_bid_list.int()
        left_up_list, right_bottom_list = collect_inside_candidate_info(inside_mask,cur_height_center_list,cur_height_half_size_list) 
        left_up_list = torch.from_numpy(left_up_list)
        right_bottom_list = torch.from_numpy(right_bottom_list)
        xmin,ymin,zmin,xmax,ymax,zmax = generate_all_conds(left_up_list,right_bottom_list)
        condx = np.bitwise_and(surf_pts[:,0] >= xmin, surf_pts[:,0] <= xmax)
        condy = np.bitwise_and(surf_pts[:,1] >= ymin, surf_pts[:,1] <= ymax)
        condz = np.bitwise_and(surf_pts[:,2] >= zmin, surf_pts[:,2] <= zmax)

        cond = np.bitwise_and(condx, condy)
        cond = np.bitwise_and(cond, condz)
        cond = cond.reshape(-1)

        if cond.sum() > 0:
            ## to get the data type of cond
            cond = cond.bool()

            box_belong[cond] = real_bid_list[cond]

    return box_belong.reshape(-1)


## sample points of specific number within the grid
def sample_points_within_grid(sample_surf_num, box_start_point, box_length, bdr_points_piece, bdr_normals_piece):
    
    ## filter points within the grid
    filter_flag = judge_pts_in_box(box_start_point, box_length, bdr_points_piece, remain_every_info = True)
    filtered_points = bdr_points_piece[filter_flag]
    filtered_normals = bdr_normals_piece[filter_flag]

    random_idx = np.random.choice(filtered_points.shape[0], sample_surf_num, replace = True)
    points = filtered_points[random_idx]
    normals = filtered_normals[random_idx]

    return points, normals



## input the list of the feature volume and the points, output the feature volume id for each point
## points: the points for judgement [N,3]
## fv_box_sp_list: the start point of the feature volume box (M,3)
## fv_box_length: the length of the feature volume box , a scalar
def get_fx_box_id_for_pts(points, fv_box_sp_list, fv_box_length):
    # Add an extra dimension to points and fv_box_sp_list for broadcasting
    points_ext = points[:, np.newaxis]
    fv_box_sp_list_ext = fv_box_sp_list[np.newaxis, :]

    # Calculate the end points of the boxes
    fv_box_ep_list = fv_box_sp_list + fv_box_length

    # Check if points are within the boxes
    in_box = np.all((points_ext >= fv_box_sp_list_ext) & (points_ext < fv_box_ep_list), axis=-1)

    # If a point is not in any box, set its box id to -1
    box_ids = np.where(in_box.any(axis=-1), in_box.argmax(axis=-1), -1)

    return box_ids



def generate_all_conds(left_up_list,right_bottom_list):

    xmin = left_up_list[:,0]
    ymin = left_up_list[:,1]
    zmin = left_up_list[:,2]
    xmax = right_bottom_list[:,0]
    ymax = right_bottom_list[:,1]
    zmax = right_bottom_list[:,2]

    return xmin,ymin,zmin,xmax,ymax,zmax

def collect_inside_candidate_info(inside_mask, cur_height_center_list, cur_height_half_size_list):

    center_piece = cur_height_center_list[inside_mask]
    half_size_piece = cur_height_half_size_list[inside_mask].reshape(-1,1)
    left_up_piece = center_piece - half_size_piece
    right_bottom_piece = center_piece +half_size_piece

    return left_up_piece, right_bottom_piece


def build_each_layer_kdtree(half_size_list,height_list,bid_list,center_list):
    
    height_list = height_list.astype(int)
    min_height = height_list.min()
    max_height = height_list.max()

    kdtree_layer_dict = {}

    for height in range(min_height,max_height+1):
        chosen_height_idx = (height_list == height).reshape(-1)
        #print(chosen_height_idx.shape)
        #print(chosen_height_idx[:5]) # [False False False False False]
        #exit(0)
        half_size_piece = half_size_list[chosen_height_idx]
        bid_piece = bid_list[chosen_height_idx]
        center_piece = center_list[chosen_height_idx].reshape(-1,3)
        kdtree_piece = cKDTree(center_piece)
        kdtree_layer_dict[height] = {
            'half_size':half_size_piece,
            'bid':bid_piece,
            'center':center_piece,
            'kdtree': kdtree_piece
        }
    
    return kdtree_layer_dict

def collect_grid_info(redboxes):
    
    half_size_list = []
    center_list = []
    bid_list = []
    height_list = []

    for bid in range(len(redboxes)):
        box = redboxes[bid]
        half_size_list.append(box.half_size)
        height_list.append(box.height)
        bid_list.append(bid)
        center_list.append(box.center)

    half_size_list = np.array(half_size_list)
    height_list = np.array(height_list)
    bid_list = np.array(bid_list)
    center_list = np.array(center_list)

    return half_size_list,height_list,bid_list,center_list
    

def write_json(content, fname):
    c = json.dumps(content)
    f2 = open(fname,'w')
    f2.write(c)
    f2.close()


def read_bin(fname):
    with open(fname, 'rb') as file:
        bin_data = file.read()
    return bin_data

def read_json(fname):
    with open(fname) as handle:
        return json.load(handle, object_hook=OrderedDict)

## generate 3d point in the bounding box
def build_grids(N, min_bound=-1, max_bound=1):
    x = np.linspace(min_bound, max_bound, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing = 'xy')
    grid_pts = np.stack((X, Y, Z), axis = -1)
    return grid_pts


def draw_colored_points_to_obj(filename, vertices, scalars_for_color=None,colormap=None,thres=None):
    print(f"draw colored points to obj file :{filename}")

    if scalars_for_color is None:
        scalars_for_color = np.zeros((vertices.shape[0],1)).reshape(-1)

    assert len(vertices.shape) == 2
    assert vertices.shape[-1] == 3
    
    if colormap == "set":
        norm = matplotlib.colors.Normalize(vmin=0, vmax=scalars_for_color.max(), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.tab20)
    else:
        if thres:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=thres, clip=True)
        else:
            norm = matplotlib.colors.Normalize(vmin=min(scalars_for_color), vmax=max(scalars_for_color), clip=True)
        #norm = matplotlib.colors.Normalize(vmin=scalars_for_color.min(), vmax=scalars_for_color.max(), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    colors = [(r, g, b) for r, g, b, a in mapper.to_rgba(scalars_for_color)]

    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    colors = [(r, g, b) for r, g, b, a in mapper.to_rgba(scalars_for_color)]
    write_obj_file(filename, vertices.reshape(-1, 3), C=colors)


## input the box information and return the updated vertices list and the edge list
## real box len: the length of the box
## real start_point: real coordinate of the point 
## draw vertices: vertices list that should be drawn
## draw edges: edges list that should be drawn
def add_grid_to_be_drawn(draw_vertices,draw_edges,real_box_len,real_start_point):

    ## draw edges
    ori_edge = np.array([
        [1, 2, 3, 4],
        [3, 4, 5, 6],
        [5, 6, 7, 8],
        [7, 8, 1, 2],
        [3, 6, 7, 2],
        [4, 5, 8, 1]
    ])
    offset_edge = ori_edge + len(draw_vertices)
    for i in range(offset_edge.shape[0]):
        draw_edges.append(list(offset_edge[i]))

    ## draw vertices
    ori_grids = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0]
    ])
    scaled_grids = real_box_len * ori_grids
    offset_grids = real_start_point + scaled_grids
    for i in range(offset_grids.shape[0]):
        draw_vertices.append(list(offset_grids[i]))
    return draw_vertices, draw_edges


def process_grids_visualization(real_start_point, real_box_len, octree = False):
    draw_vertices = []
    draw_edges = []
    for point_idx in range(real_start_point.shape[0]):
        if octree:
            cur_box_len = real_box_len[point_idx]
        else:
            cur_box_len = real_box_len
        draw_vertices,draw_edges = add_grid_to_be_drawn(draw_vertices, draw_edges, cur_box_len, real_start_point[point_idx])
    return draw_vertices, draw_edges



def write_obj_file(filename, V, F=None, C=None, vid_start=1):
    with open(filename, 'w') as f:
        if C is not None:

            for Vi, Ci in zip(V, C):
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]} {Ci[0]} {Ci[1]} {Ci[2]}\n")
        else:
            for Vi in V:
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]}\n")

        if F is not None:
            for Fi in F:
                if len(Fi)==3:
                    f.write(f"f {Fi[0]+vid_start} {Fi[1]+vid_start} {Fi[2]+vid_start}\n")
                else:
                    f.write(f"f {Fi[0]+vid_start} {Fi[1]+vid_start} {Fi[2]+vid_start} {Fi[3]+vid_start}\n")



def image_to_np(img):
    return np.array(img).transpose(2,0,1)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colorize_time(elapsed):
    if elapsed > 1e-3:
        return bcolors.FAIL + "{:.3e}".format(elapsed) + bcolors.ENDC
    elif elapsed > 1e-4:
        return bcolors.WARNING + "{:.3e}".format(elapsed) + bcolors.ENDC
    elif elapsed > 1e-5:
        return bcolors.OKBLUE + "{:.3e}".format(elapsed) + bcolors.ENDC
    else:
        return "{:.3e}".format(elapsed)

class PerfTimer():
    def __init__(self, activate=False):
        self.prev_time = time.process_time()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.prev_time_gpu = self.start.record()
        self.counter = 0
        self.activate = activate

    def reset(self):
        self.counter = 0
        self.prev_time = time.process_time()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.prev_time_gpu = self.start.record()

    def check(self, name=None):
        if self.activate:
            cpu_time = time.process_time() - self.prev_time
            cpu_time = colorize_time(cpu_time)
          
            self.end.record()
            torch.cuda.synchronize()

            gpu_time = self.start.elapsed_time(self.end) / 1e3
            gpu_time = colorize_time(gpu_time)
            if name:
                print("CPU Checkpoint {}: {} s".format(name, cpu_time))
                print("GPU Checkpoint {}: {} s".format(name, gpu_time))
            else:
                print("CPU Checkpoint {}: {} s".format(self.counter, cpu_time))
                print("GPU Checkpoint {}: {} s".format(self.counter, gpu_time))

            self.prev_time = time.process_time()
            self.prev_time_gpu = self.start.record()
            self.counter += 1
            return cpu_time, gpu_time


def sample_box_boundary(start_p,box_length,box_edge_num,box_plane_num):
    
    ## sample edge points
    edge_pts = []
    edge_sp_list,edge_add_list = edge_info_cube(start_p,box_length)
    for i in range(12):
        edge_sp = edge_sp_list[i]
        edge_add = edge_add_list[i]
        edge_pts.append(sample_edge(edge_sp,edge_add,int(box_edge_num/12),box_length))
    edge_pts = np.concatenate(edge_pts)

    ## sample plane points
    plane_pts = []
    plane_sp_list,plane_add_list = plane_info_cube(start_p,box_length)
    box_plane_axis_num = int(math.sqrt(box_plane_num/6))
    for i in range(6):
        plane_sp = plane_sp_list[i]
        plane_add = plane_add_list[i]
        plane_pts_piece = sample_plane(plane_sp,plane_add,box_plane_axis_num,box_length)
        plane_pts.append(plane_pts_piece)
    plane_pts = np.concatenate(plane_pts)

    return edge_pts,plane_pts

def edge_info_cube(start_p,box_length):
    sp_list = [
        [0,0,0],[0,0,0],[0,0,0],
        [1,1,0],[1,1,0],[1,1,0],
        [0,1,1],[0,1,1],[0,1,1],
        [1,0,1],[1,0,1],[1,0,1]
    ]

    add_list = [
        [0,1,0],[1,0,0],[0,0,1],
        [-1,0,0],[0,-1,0],[0,0,1],
        [1,0,0],[0,0,-1],[0,-1,0],
        [0,0,-1],[-1,0,0],[0,1,0]
    ]

    sp_list = start_p + np.array(sp_list) * box_length
    add_list = np.array(add_list)

    return sp_list, add_list

def plane_info_cube(start_p,box_length):
    sp_list = [
        [0,0,0],[0,0,0],[0,0,0],
        [1,1,1],[1,1,1],[1,1,1]
    ]
    add_list = [
        [1,1,0],[1,0,1],[0,1,1],
        [0,-1,-1],[-1,-1,0],[-1,0,-1]
    ]

    sp_list = start_p + np.array(sp_list) * box_length
    add_list = np.array(add_list)

    return sp_list,add_list

def sample_edge(sp,add,sample_num,box_length):
    sp = sp.reshape(-1,3)
    edge_pts = sp.repeat(sample_num,axis = 0)

    if abs(add[0]) == 1:
        x = np.linspace(0, add[0] * box_length, sample_num).reshape(-1)
        edge_pts[:,0] = edge_pts[:,0] + x

    elif abs(add[1]) == 1:
        x = np.linspace(0, add[1] * box_length, sample_num).reshape(-1)
        edge_pts[:,1] = edge_pts[:,1] + x

    elif abs(add[2]) == 1:
        x = np.linspace(0, add[2] * box_length, sample_num).reshape(-1)
        edge_pts[:,2] = edge_pts[:,2] + x

    else:
        print('Wrong add list for edge sampling')
        exit(0)
    
    return edge_pts

def sample_plane(sp,add,sample_num,box_length):
    x_list = []
    for j in range(3):
        if abs(add[j]) == 1:
            tempo_x = np.linspace(0, add[j] * box_length, sample_num)
            x_list.append(tempo_x)
        if add[j] == 0:
            tempo_x = np.linspace(0, 0, 1)
            x_list.append(tempo_x)

    X, Y, Z = np.meshgrid(x_list[0], x_list[1],x_list[2], indexing='xy')
    plane_pts = np.stack((X, Y, Z), axis=-1)
    plane_pts = plane_pts.reshape(-1,3)
    plane_pts = plane_pts + sp
    return plane_pts


def replace_bdr_pts(filter_pts, box_sp, box_length):
    box_end = box_sp + box_length
    # Expand dimensions for broadcasting
    box_end_exp = box_end.unsqueeze(1)
    filter_pts_exp = filter_pts.unsqueeze(0)
    # Find matching lines
    matches = torch.all(box_end_exp == filter_pts_exp, dim=-1)
    # Change matching lines in filter_pts
    filter_pts[torch.any(matches, dim=0)] = filter_pts[torch.any(matches, dim=0)] * 0.999  # Replace with your own logic
    return filter_pts

''''
def sample_box(box_sp, box_length, sample_num):

    box_end = box_sp + box_length
    x = np.linspace(box_sp[0], box_end[0], sample_num)
    y = np.linspace(box_sp[1], box_end[1], sample_num)
    z = np.linspace(box_sp[2], box_end[2], sample_num)
    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')
    sample_pts = np.stack((X, Y, Z), axis=-1)
    sample_pts = sample_pts.reshape(-1, 3)

    return sample_pts

def sample_pts_in_merge_grids(box_arr, box_len, sample_num_per_box):
    ## box_arr: the left up points of the boxes, [N,3]
    ## box_len: the length of each box, [N,1]
    ## sample_num_per_box: the number of points to be sampled in each box, a scalar
    scale_num = 0.999
    box_num = box_arr.shape[0]
    sample_pts = []
    for i in range(box_num):
        sample_pts.append(sample_box(box_arr[i], box_len[i] * scale_num, sample_num_per_box))
    sample_pts = np.concatenate(sample_pts)
    return sample_pts
'''

## random sample within the box
def random_sample_box(box_sp, box_length, sample_num):
    #print(box_sp.device)
    #print(box_length.device)
    #exit(0)
    sample_pts = torch.rand(sample_num, 3).to(box_length.device) * box_length + box_sp
    return sample_pts


def sample_pts_in_merge_grids_new(box_arr, box_len, sample_num_per_box,scale_facotr): #, sample_more_length):

    scale_num = 0.999
    box_num = box_arr.shape[0]
    #box_len = box_len + 2 * sample_more_length
    #box_arr = box_arr - sample_more_length
    sample_pts = []
    for i in range(box_num):
        sample_pts.append(random_sample_box(box_arr[i], box_len[i] * scale_num, sample_num_per_box))
    sample_pts = torch.cat(sample_pts)
    return sample_pts

## given array of box_start_points and array of box_length, and also the extend_box_length, return the points in the boxes
def filter_points_in_boxes(box_arr, box_len, pts, extend_box_length):

    box_arr = box_arr - extend_box_length
    box_len = box_len + 2 * extend_box_length

    # Calculate the coordinates of the opposite corner of each box
    box_end = box_arr + box_len

    # Assuming box_arr, box_end, and pts are tensors
    box_arr_exp = box_arr.unsqueeze(1)
    box_end_exp = box_end.unsqueeze(1)
    pts_exp = pts.unsqueeze(0)

    # Create a boolean mask that identifies which points are inside each box
    mask = (pts_exp >= box_arr_exp) & (pts_exp < box_end_exp)
    mask = torch.all(mask, dim=-1)
    mask = torch.any(mask, dim=0).view(-1)
    
    pts_in_boxes = pts[mask]

    return pts_in_boxes, mask






