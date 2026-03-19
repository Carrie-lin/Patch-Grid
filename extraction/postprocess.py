from glob import glob
import os
import trimesh
import torch
import numpy as np
from tqdm import tqdm
import math

from utils import write_obj_file, return_box_cutted_mesh, get_box_id_for_pts, draw_colored_points_to_obj, judge_pts_in_box, read_json
from extraction.extractutils import get_predefined_redbox
from preparation.lib.graph import define_merge_order

import subprocess
import sys
import pathlib
import open3d as o3d


class PatchBelong:
    def __init__(self) -> None:
        # get the current working directory absolute path
        self.bin_dir = pathlib.Path(__file__).parent.absolute() / "bin"
        self.cwd = pathlib.Path(__file__).parent.parent.absolute()
        # if windows
        exe = "extraction/patch_split/build/split_patch"
        if sys.platform == "win32":
            self.exe = exe + ".exe"
        # if linux
        elif sys.platform == "linux":
            self.exe = exe
        else:
            raise Exception("OS not supported")
        # add the exe to the environment path
        os.environ["PATH"] += os.pathsep + str(self.bin_dir)
        os.chdir(self.cwd)
        pass

    def call(self, input_dir, output_dir):
        command = [ str(self.exe), 
                    str(input_dir),
                    str(output_dir)]
        print('Splitting the patches according to the intersections...')
        subprocess.run(command, shell=False)

class MeshInter:
    def __init__(self) -> None:
        # get the current working directory absolute path
        self.bin_dir = pathlib.Path(__file__).parent.absolute() / "bin"
        self.cwd = pathlib.Path(__file__).parent.parent.absolute()
        # if windows
        exe = "extraction/FastAndRobustMeshArrangements/build/mesh_arrangement"
        if sys.platform == "win32":
            self.exe = exe + ".exe"
        # if linux
        elif sys.platform == "linux":
            self.exe = exe
        else:
            raise Exception("OS not supported")
        # add the exe to the environment path
        os.environ["PATH"] += os.pathsep + str(self.bin_dir)
        os.chdir(self.cwd)

        pass

    def call(self, input_file, output_file):
        command = [ str(self.exe), 
                    str(input_file),
                    str(output_file)]
        print('Computing the intersection of the patches...')
        subprocess.run(command, shell=False)


## ori_shapes and new_shapes are the same trimesh meshes, but with different size, we want to align them
def align_shapes(ori_shape, new_shape):
   ## get the bounding box of ori_shape
    ori_bbox = ori_shape.bounding_box_oriented
    ori_center = (ori_bbox.transform)[:3,3].reshape(-1)
    ori_size = ori_bbox.extents

    ## get the bounding box of new_shape
    new_bbox = new_shape.bounding_box_oriented
    new_center = (new_bbox.transform)[:3,3].reshape(-1)
    new_size = new_bbox.extents

    scale_factor = (ori_size/new_size)[0]

    ## scale the new_shape
    new_shape.apply_scale(scale_factor)

    ## transform the new_shape so that it would have the same center as the ori_shape
    new_shape.apply_translation(ori_center - new_center)

    ## export new shape
    return new_shape

def recompose_mesh_patches(save_dir, geometric_prefix, exp_name, cutted_flag = False):

    fnames = glob(os.path.join(save_dir, "composed_*.obj"))
    print("Total patches --- num files", len(fnames))

    vs = []
    fs = []
    start_num = 0
    for i, f in enumerate(fnames):
        patch = trimesh.load(f)
        v = patch.vertices
        f = patch.faces
        vs.append(v)
        fs.append(f+start_num)
        start_num += len(v)

    if len(vs) > 0:
        vs = np.concatenate(vs, axis=0)
        fs = np.concatenate(fs, axis=0)
        if cutted_flag:
            file_name = f"{exp_name}_composed_all.obj"
        else:
            file_name = f"{exp_name}_composed_all_no_cut.obj"
        write_obj_file(os.path.join(geometric_prefix, file_name), V=vs, F=fs)
        print(os.path.join(geometric_prefix, file_name))

    else:
        print("Warning: vs is empty")

    return 


def recompose_mesh(save_dir, patch_id=None, rm=True, geometric_prefix=None, exp_name = None, cutted_flag = True):

    if patch_id is None:
        if cutted_flag:
            fnames = glob(os.path.join(save_dir, "cutted_mc_surf_pts_*.obj"))
        else:
            fnames = glob(os.path.join(save_dir, "mc_surf_pts_*.obj"))
    else:
        assert type(patch_id) is int
        fnames = glob(os.path.join(save_dir, f"original_mc_surf_pts_{patch_id}_*.obj"))

    print(patch_id, " --- num files", len(fnames))

    vs = []
    fs = []
    start_num = 0
    for i, f in enumerate(fnames):
        patch = trimesh.load(f)
        v = patch.vertices
        f = patch.faces
        vs.append(v)
        fs.append(f+start_num)
        start_num += len(v)

    if len(vs) > 0:
        vs = np.concatenate(vs, axis=0)
        fs = np.concatenate(fs, axis=0)
        ## merge close points

        if patch_id is None:
            write_obj_file(os.path.join(geometric_prefix,f"{exp_name}_composed_all.obj"), V=vs, F=fs)
            print(os.path.join(geometric_prefix,f"{exp_name}_composed_all.obj"))
        else:
            file_name = os.path.join(save_dir,f"composed_{patch_id}.obj")
            write_obj_file(file_name, V=vs, F=fs)
            patch_mesh = o3d.io.read_triangle_mesh(file_name, enable_post_processing=False, print_progress=False)
            patch_mesh.merge_close_vertices(2/10000 * math.sqrt(4+4+4))
            o3d.io.write_triangle_mesh(file_name, patch_mesh, write_triangle_uvs=False, write_vertex_colors=False, print_progress=False)

    else:
        print("Warning: vs is empty")

    if not(patch_id is None) and rm:
        ## remove all the files in the save_dir that contains mc_surf_pts
        fnames1 = glob(os.path.join(save_dir, "cutted_mc_surf_pts_*.obj"))
        fnames2 = glob(os.path.join(save_dir, "mc_surf_pts_*.obj"))
        fnames3 = glob(os.path.join(save_dir, f"original_mc_surf_pts_{patch_id}_*.obj"))
        for f in fnames1:
            os.remove(f)
        for f in fnames2:
            os.remove(f)
        for f in fnames3:
            os.remove(f)
       
    print("done")

def cut_extended_meshes(save_dir, geometric_prefix, dataset_path, exp_name, cut_boundary_length):
    ## load extraction box

    ## get merge boxes
    file_redboxes = f'{dataset_path}/redboxes_extract.json'
    redboxes = get_predefined_redbox(file_redboxes)

    extract_box_info = {}

    for bid, rbx in enumerate(redboxes):
        extract_box_info[bid] = {
            'center': rbx.center,
            'half_size': rbx.half_size,
        }
    
    fnames = glob(os.path.join(save_dir, "mc_surf_pts_*.obj"))
    for i, f in enumerate(fnames):
        patch = trimesh.load(f)

        ## get box id from the name of the file
        file_name = f.split('/')[-1]
        number_list = file_name.replace('mc_surf_pts_','')
        number_list = number_list.replace('.obj','')
        bid = number_list.split('_')[1]

        ## get start points and box length of each box
        start_points = extract_box_info[int(bid)]['center'] - extract_box_info[int(bid)]['half_size']
        box_length = extract_box_info[int(bid)]['half_size'] * 2

        ## update the start point and box length with cut_boundary_length
        start_points = start_points - cut_boundary_length
        box_length = box_length + 2 * cut_boundary_length

        ## cut the patch
        new_patch = return_box_cutted_mesh(patch, start_points, box_length)

        if new_patch is None:
            continue
        
        ## save the new patch with new name
        new_patch_name = f.replace('mc_surf_pts_','cutted_mc_surf_pts_')
        new_patch.export(new_patch_name)

    return 

def connect_all_meshes():
    return 

def adjust_grid_points_with_append_layer(append_layer_num, rbx, N):
    if append_layer_num > 0:
        ori_step_len = (2 * rbx.half_size) / (N - 1)
        N = 2 * append_layer_num + N
        adjust_half_size = append_layer_num * ori_step_len + rbx.half_size
    else:
        adjust_half_size = rbx.half_size 
        N = N
    return N, adjust_half_size


def load_precomputed(dataset_path, sdf_thres):
    shape_name = dataset_path.split('/')[-1]
    file_name = f'debug/{shape_name}_sdf_mean.txt'
    save_info = np.loadtxt(file_name)
    sub_mesh_id = save_info[:,0]
    sdf_mean_list = save_info[:,1]
    remain_flag_list = []
    for i in range(int(sub_mesh_id.max())+1):
        print(f'Sub mesh {i} | Mean SDF:', sdf_mean_list[i])
        if sdf_mean_list[i] >= sdf_thres:
            remain_flag_list.append(False)
        else:
            remain_flag_list.append(True)
    #save_info = np.stack([np.arange(int(sub_mesh_id.max())+1), np.array(sdf_mean_list)], axis=1)
    #np.savetxt(file_name, save_info)
    return remain_flag_list
    

def wrap_geometry(geo_path):
    patch_mesh = o3d.io.read_triangle_mesh(geo_path, enable_post_processing=False, print_progress=False)
    patch_mesh.merge_close_vertices(5/10000 * math.sqrt(4+4+4))
    o3d.io.write_triangle_mesh(geo_path, patch_mesh, write_triangle_uvs=False, write_vertex_colors=False, print_progress=False)


def back_to_field_check(vs,
                        sub_mesh_id,               
                        open,
                        net,
                        save_prefix,
                        dataset_path,
                        extract_resolution,
                        device,
                        vis_sdf,
                        vis_belong,
                        append_layer_num,
                        query_batch_size,
                        cut_boundary_length,
                        sdf_thres):

    vs_refer = vs.copy()
    vs = torch.tensor(vs, dtype=torch.float32)

    ## get extraction red boxes
    file_redboxes = f'{dataset_path}/redboxes_extract.json'
    redboxes = get_predefined_redbox(file_redboxes)

    ## get edge types
    edge_types = np.loadtxt(os.path.join(dataset_path, 'connect_type.txt'))
    
    ## judge which box these points belong to 
    box_id = get_box_id_for_pts(vs, redboxes)

    ## get the box information according to the box id list
    vs = vs.to(device)
    sdf_value = get_sdf_from_our_net(vs, vs_refer, net, redboxes, query_batch_size, edge_types, sdf_thres, append_layer_num, extract_resolution)
    remain_flag_list = []

    shape_name = dataset_path.split('/')[-1]
    #draw_colored_points_to_obj(f'debug/{shape_name}_err.obj',vs, np.abs(sdf_value).reshape(-1))

    sdf_mean_list = []

    for i in range(int(sub_mesh_id.max())+1):

        mask = (sub_mesh_id == i).reshape(-1)
        cur_sub_mesh_sdf = sdf_value[mask]
        cur_sub_mesh_sdf_mean = np.mean(np.abs(cur_sub_mesh_sdf))
        sdf_mean_list.append(cur_sub_mesh_sdf_mean)
        print(f'Sub mesh {i} | Mean SDF:', cur_sub_mesh_sdf_mean)
        if cur_sub_mesh_sdf_mean >= sdf_thres:
            print('Throw away patch')
            remain_flag_list.append(False)
        else:
            print('Remain patch.')
            remain_flag_list.append(True)
    
    ## save sdf mean list and its corresponding sub mesh id 
    #save_info = np.stack([np.arange(int(sub_mesh_id.max())+1), np.array(sdf_mean_list)], axis=1)
    #np.savetxt(f'debug/{shape_name}_sdf_mean.txt', save_info)

    return remain_flag_list

def get_sdf_from_our_net(all_pts, all_pts_refer, network, redboxes, query_batch_size, edge_types, sdf_thres, append_layer, extract_resolution):

    ## prepare the array to store the sdf value
    sdf_list = np.ones((all_pts.shape[0], 1), dtype=np.float32) * sdf_thres * 2
    visited_flag = np.zeros((all_pts.shape[0], 1), dtype=np.float32)

    ## go over each merge box
    for bid, rbx in tqdm(enumerate(redboxes)):

        assert len(rbx.fids) > 0
        
        ## get the points with larger bounding box
        N = int(extract_resolution / (2 ** (rbx.height))) + 1
        _, adjust_half_size = adjust_grid_points_with_append_layer(append_layer, rbx, N)
        start_pts = rbx.center - adjust_half_size
        mask = judge_pts_in_box(start_pts, adjust_half_size * 2, all_pts_refer, remain_every_info = True).reshape(-1)
        pts = all_pts[mask]
        
        if pts.shape[0]<=1:
            continue

        if len(rbx.fids) == 1:

            ## single patch
            pid = rbx.fids[0]
            pnts_iterator = enumerate(torch.split(pts, query_batch_size, dim=0))

            z = []
            for i, pnts in pnts_iterator:
                z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                z.append(z_p)
            z = torch.cat(z,dim=0)
            z = z.numpy().astype(np.float32)
        
        else:

            merge_order, sorted_fids, duplicate_box = define_merge_order(rbx.fids, edge_types)

            """
            return type:
            0: len(idx) = 0; in this case, input are some patches that are not connected in the space
            1: len(fids) > len(idx); in this case, some disconnected patches are missed while other connected merged.
            2: number of nodes after merged larger than 1; 
                - in this case, there are more than 1 connected components in the grid (rarely seen)
            -1: all patches are merged into a single connected component
            """
            if duplicate_box == 0:

                assert len(sorted_fids) == 0
                z_list = []
                for pid in rbx.fids:
                    pnts_iterator = enumerate(torch.split(pts, query_batch_size, dim=0))

                    z = []
                    for i, pnts in pnts_iterator:
                        z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                        z.append(z_p)
                    z = torch.cat(z,dim=0)
                    z = z.numpy().astype(np.float32)
                    z_list.append(z)
                z_list = np.stack(z_list, axis=1)
                ## get abs of all elements in z_list
                z_list = np.abs(z_list)
                z = np.min(z_list, axis=1)

            elif duplicate_box == 1:

                ## processing the merged patches
                sdf_qs = []
                for pid in sorted_fids:
                    z = []

                    pnts_iterator = enumerate(torch.split(pts, query_batch_size, dim=0))
                    
                    z = []
                    for i, pnts in pnts_iterator:
                        z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                        z.append(z_p)
                    z = torch.cat(z,dim=0)
                    z = z.numpy().astype(np.float32)

                    sdf_qs.append(z)

                sdf_qs = np.stack(sdf_qs, axis=0)
                sdf_belong = np.ones_like(sdf_qs) * (-1)
                kept_ids = set(np.arange(len(sorted_fids)))

                for how_to_merge in merge_order:
                    to_id = how_to_merge["edge"][0]
                    from_id = how_to_merge["edge"][1]
                    kept_ids.discard(from_id)
                    out = sdf_qs[[to_id, from_id],...]
                    if how_to_merge["property"] == 1:
                        out_index = np.argmax(out, axis=0)
                        out = np.max(out, axis=0) 
                    if how_to_merge["property"] == -1:
                        out_index = np.argmin(out, axis=0)
                        out = np.min(out, axis=0)
                    sdf_qs[to_id] = out
                    sdf_belong[to_id] = out_index

                z_list = []
                for idx in kept_ids:
                    z = sdf_qs[idx]
                    belong_z = sdf_belong[idx]
                    z_list.append(z)

                    assert belong_z.min() >= 0

                ## processing the unmerged patches as individual patches
                left_ids = set(rbx.fids).difference(set(sorted_fids))
                for pid in left_ids:
                    pnts_iterator = enumerate(torch.split(pts, query_batch_size, dim=0))

                    z = []
                    for i, pnts in pnts_iterator:
                        z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                        z.append(z_p)
                    z = torch.cat(z,dim=0)
                    z = z.numpy().astype(np.float32)
                    z_list.append(z)
                
                z_list = np.stack(z_list, axis=1)
                z_list = np.abs(z_list)
                z = np.min(z_list, axis=1)
    
            ## -1: all patches are merged into a single connected component
            elif duplicate_box == -1:

                sdf_qs = []
                for pid in sorted_fids:
                    z=[]
                    pnts_iterator = enumerate(torch.split(pts, query_batch_size, dim=0))

                    z = []
                    for i, pnts in pnts_iterator:
                        z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                        z.append(z_p)
                    z = torch.cat(z,dim=0)
                    z = z.numpy().astype(np.float32)

                    sdf_qs.append(z)

                sdf_qs = np.stack(sdf_qs, axis=0) 
                sdf_belong = np.ones_like(sdf_qs) * (-1)
                kept_ids = set(np.arange(len(sorted_fids)))

                for how_to_merge in merge_order:
                    to_id = how_to_merge["edge"][0]
                    from_id = how_to_merge["edge"][1]
                    kept_ids.discard(from_id)
                    out = sdf_qs[[to_id, from_id],...]

                    if how_to_merge["property"] == 1:
                        out_index = np.argmax(out.reshape, axis=0)
                        out = np.max(out, axis=0)
                
                    if how_to_merge["property"] == -1:
                        out_index = np.argmin(out, axis=0)
                        out = np.min(out, axis=0)
                    sdf_qs[to_id] = out
                    sdf_belong[to_id] = out_index
                kept_ids = list(kept_ids)
                assert len(kept_ids) == 1
                z = sdf_qs[to_id]
                belong_z = sdf_belong[to_id]

                assert belong_z.min() >= 0 
 
            ## number of nodes after merged larger than 1
            elif duplicate_box == 2:
                cnt = 0
                z_list = []
                for m_order, new_sorted_ids  in zip(merge_order, sorted_fids):


                    cnt += 1
                    sdf_qs = []
                    for pid in new_sorted_ids:
                        z=[]
                        pnts_iterator = enumerate(torch.split(pts, query_batch_size, dim=0))

                        z = []
                        for i, pnts in pnts_iterator:
                            z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                            z.append(z_p)
                        z = torch.cat(z,dim=0)
                        z = z.numpy().astype(np.float32)

                        sdf_qs.append(z)
        
                    sdf_qs = np.stack(sdf_qs, axis=0) ## [K, N, 1]
                    sdf_belong = np.ones_like(sdf_qs) * (-1)
                    kept_ids = set(np.arange(len(new_sorted_ids)))

                    for how_to_merge in m_order:
                        to_id = how_to_merge["edge"][0]
                        from_id = how_to_merge["edge"][1]
                        kept_ids.discard(from_id)
                        out = sdf_qs[[to_id, from_id],...]
                        if how_to_merge["property"] == 1:
                            out_index = np.argmax(out, axis=0)
                            out = np.max(out, axis=0)
               
                        if how_to_merge["property"] == -1:
                            out_index = np.argmin(out, axis=0)
                            out = np.min(out, axis=0)
                        sdf_qs[to_id] = out
                        sdf_belong[to_id] = out_index

                    kept_ids = list(kept_ids)
                    assert len(kept_ids) == 1
                    z = sdf_qs[to_id]
                    belong_z = sdf_belong[to_id]
                    assert belong_z.min() >= 0 
                    z_list.append(z)

                ## processing the unmerged patches as individual patches
                flatten_sorted_fids = [item for sublist in sorted_fids for item in sublist]
                left_ids = set(rbx.fids).difference(set(flatten_sorted_fids))
                for pid in left_ids:
                    pnts_iterator = enumerate(torch.split(pts, query_batch_size, dim=0))

                    z = []
                    for i, pnts in pnts_iterator:
                        z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                        z.append(z_p)
                    z = torch.cat(z,dim=0)
                    z = z.numpy().astype(np.float32)
                    z_list.append(z)
                
                z_list = np.stack(z_list, axis=1)
                z_list = np.abs(z_list)
                z = np.min(z_list, axis=1)
    
            else:
                raise NotImplementedError   
            
        ## replace the sdf array at the mask palces  with the new sdf value z only if the z is smaller than the current sdf value
        ori_z = sdf_list[mask]
        replace_flag = ((np.abs(z)).reshape(-1) < (np.abs(ori_z)).reshape(-1)).reshape(-1)
        ori_z[replace_flag] = z[replace_flag].reshape(-1,1)
        sdf_list[mask] = ori_z
        
    return sdf_list


def re_normalize_small_items(save_dir, exp_name, dataset_path):    

    ## judge whether there is a file named scale_trans.json
    scale_trans_file_path = f'{dataset_path}/scale_trans.json'
    if os.path.exists(scale_trans_file_path):

        print('Scale the small items to its original size...')

        ## load scale and translate information
        scale_trans_info = read_json(scale_trans_file_path)
        scale_factor = scale_trans_info['scale']
        translate_value = scale_trans_info['translate']
        center = scale_trans_info['center']

        ## load the mesh
        mesh = trimesh.load(os.path.join(save_dir, f'{exp_name}_composed_all.obj'))

        mesh.vertices = mesh.vertices * scale_factor
        mesh.vertices = mesh.vertices - translate_value
        mesh.vertices = mesh.vertices + center

        mesh.export(os.path.join(save_dir, f'{exp_name}_scaled.obj'))

    else:
        return 
    
