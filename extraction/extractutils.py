import os
import sys
sys.path.append(os.getcwd())

from tqdm import tqdm
from skimage import measure
import shutil
import torch
import numpy as np
import torch.optim.lr_scheduler

from preparation.lib.mergeutils import RedBox
from preparation.lib.graph import define_merge_order
from utils import draw_colored_points_to_obj, write_obj_file, read_json

def marching_cube(volume, scale, t, savefilename, offset = 0):
    try:
        v, faces, _, _ = measure.marching_cubes_lewiner(volume, offset)
        v = (v*scale) + t
        faces = faces[:,[0,2,1]]
        write_obj_file(savefilename, v, F=faces)
    except ValueError:
        #print('The grid does not extract any geometry.')
        a = 1

def get_predefined_redbox(file_redboxes):
    redboxes_ori = read_json(file_redboxes)
    redboxes = []
    for rbx in redboxes_ori:
        half_size = rbx['half_size'] 
        rbxx = RedBox(np.array(rbx['center']), half_size)
        rbxx.set_face_ids(rbx['face_ids'])
        rbxx.set_height(rbx['height'])
        redboxes.append(rbxx)
    return redboxes

def subtle_process_for_pts(global_q):
    detect_boundary_flag = (global_q >= 1)
    global_q[detect_boundary_flag] = 0.999
    return global_q

def adjust_grid_points_with_append_layer(append_layer_num, rbx, N):
    if append_layer_num > 0:
        ori_step_len = (2 * rbx.half_size) / (N - 1)
        N = 2 * append_layer_num + N
        adjust_half_size = append_layer_num * ori_step_len + rbx.half_size
    else:
        adjust_half_size = rbx.half_size 
        N = N
    return N, adjust_half_size

def plot_rbx_marchingcube(network,
                          save_dir,
                          dataset_path, 
                          extract_resolution,
                          device,
                          edge_types,
                          vis_sdf,
                          vis_belong,
                          append_layer_num,
                          extract_batch_size, 
                          sdf_thres = 0.1,
                          delete_patch_list = None):
    
    ## create a new folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    ## get merge boxes
    file_redboxes = f'{dataset_path}/redboxes_extract.json'
    redboxes = get_predefined_redbox(file_redboxes)

    ## go over each merge box
    for bid, rbx in tqdm(enumerate(redboxes)):

        assert len(rbx.fids) > 0

        ## generate the sample points
        assert extract_resolution >= 2**rbx.height
        N = int(extract_resolution / (2 ** (rbx.height))) + 1
        N, adjust_half_size = adjust_grid_points_with_append_layer(append_layer_num, rbx, N)

        rbx.build_box(N, custom_half_size = adjust_half_size) 
        scale = 2 * adjust_half_size / (N - 1)
        t = rbx.center - np.array([[adjust_half_size, adjust_half_size, adjust_half_size]])
        grid_pts = rbx.get_grid_pts() 
        global_q = torch.FloatTensor(grid_pts).to(device)
        global_q = subtle_process_for_pts(global_q)

        ## delete the patches that are in the delete patch list
        if not (delete_patch_list is None):
            ## remove the fids in rbx.fids that are in delete_patch_list
            rbx.fids = [item for item in rbx.fids if item not in delete_patch_list]
            if len(rbx.fids) == 0:
                continue

        if len(rbx.fids) == 1:

            ## single patch
            pid = rbx.fids[0]
            local_q = global_q
            pnts_iterator = enumerate(torch.split(local_q, extract_batch_size, dim=0))

            z = []
            for i, pnts in pnts_iterator:
                z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                z.append(z_p)
            z = torch.cat(z,dim=0)
            z = z.numpy().astype(np.float32)

            savefilename = os.path.join(save_dir, f"mc_surf_pts_{pid}_{bid}.obj")  
            marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)
            savefilename = os.path.join(save_dir, f"original_mc_surf_pts_{pid}_{bid}.obj")
            marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)

            if vis_sdf:
                draw_colored_points_to_obj(
                                os.path.join(save_dir, f"single_grid_pts_{bid}_{pid}_fused.obj"), 
                                vertices=grid_pts.reshape(-1,3), 
                                scalars_for_color = z.squeeze(), colormap='jet',thres = sdf_thres)
        else:
            ## multiple patches 
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
                for pid in rbx.fids:
                    local_q = global_q
                    pnts_iterator = enumerate(torch.split(local_q, extract_batch_size, dim=0))

                    z = []
                    for i, pnts in pnts_iterator:
                        z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                        z.append(z_p)
                    z = torch.cat(z,dim=0)
                    z = z.numpy().astype(np.float32)

                    savefilename = os.path.join(save_dir, f"mc_surf_pts_{pid}_{bid}.obj")
                    marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)
                    savefilename = os.path.join(save_dir, f"original_mc_surf_pts_{pid}_{bid}.obj")
                    marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)

                    if vis_sdf:
                        draw_colored_points_to_obj(
                            os.path.join(save_dir, f"merge_grid_pts_{bid}_{pid}_fused.obj"), 
                            vertices = grid_pts.reshape(-1,3), 
                            scalars_for_color = z.squeeze(), colormap='jet',thres = sdf_thres)

            elif duplicate_box == 1:
                
                #continue
                
                ## processing the merged patches
                sdf_qs = []
                for pid in sorted_fids:
                    z = []
                    local_q = global_q

                    pnts_iterator = enumerate(torch.split(local_q, extract_batch_size, dim=0))
                    
                    z = []
                    for i, pnts in pnts_iterator:
                        z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                        z.append(z_p)
                    z = torch.cat(z,dim=0)
                    z = z.numpy().astype(np.float32)

                    sdf_qs.append(z)
                    savefilename = os.path.join(save_dir, f"original_mc_surf_pts_{pid}_{bid}.obj")
                    marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)

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

                for idx in kept_ids:
                    z = sdf_qs[idx]
                    belong_z = sdf_belong[idx]

                    assert belong_z.min() >= 0

                    savefilename = os.path.join(save_dir, f"mc_surf_pts_{sorted_fids[idx]}_{bid}.obj")
                    marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)

                    if vis_sdf:
                        draw_colored_points_to_obj(
                            os.path.join(save_dir, f"merge_grid_pts_{bid}_fused.obj"), 
                            vertices = grid_pts.reshape(-1,3), 
                            scalars_for_color = z.squeeze(), colormap='jet',thres = sdf_thres)
                        
                    if vis_belong:
                        draw_colored_points_to_obj(
                            os.path.join(save_dir, f"merge_pts_belong_{bid}.obj"), 
                            vertices = grid_pts.reshape(-1,3), 
                            scalars_for_color = belong_z.squeeze(), colormap='set')
            
                ## processing the unmerged patches as individual patches
                left_ids = set(rbx.fids).difference(set(sorted_fids))
                for pid in left_ids:
                    local_q = global_q
                    pnts_iterator = enumerate(torch.split(local_q, extract_batch_size, dim=0))

                    z = []
                    for i, pnts in pnts_iterator:
                        z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                        z.append(z_p)
                    z = torch.cat(z,dim=0)
                    z = z.numpy().astype(np.float32)

                    savefilename = os.path.join(save_dir, f"mc_surf_pts_{pid}_{bid}.obj")
                    marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)
                    savefilename = os.path.join(save_dir, f"original_mc_surf_pts_{pid}_{bid}.obj")
                    marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)
                    if vis_sdf:
                        draw_colored_points_to_obj(
                            os.path.join(save_dir, f"merge_grid_pts_{bid}_{pid}_fused.obj"), 
                            vertices=grid_pts.reshape(-1,3), 
                            scalars_for_color=z.squeeze(), colormap='jet',thres = sdf_thres)
                        
            ## -1: all patches are merged into a single connected component
            elif duplicate_box == -1:

                sdf_qs = []
                for pid in sorted_fids:
                    z=[]
                    local_q = global_q
                    pnts_iterator = enumerate(torch.split(local_q, extract_batch_size, dim=0))

                    z = []
                    for i, pnts in pnts_iterator:
                        z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                        z.append(z_p)
                    z = torch.cat(z,dim=0)
                    z = z.numpy().astype(np.float32)

                    sdf_qs.append(z)
                    savefilename = os.path.join(save_dir, f"original_mc_surf_pts_{pid}_{bid}.obj")
                    marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)

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

                savefilename = os.path.join(save_dir, f"mc_surf_pts_{sorted_fids[kept_ids[0]]}_{bid}.obj") 
                marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)

                if vis_sdf:
                    draw_colored_points_to_obj(
                        os.path.join(save_dir, f"merge_grid_pts_{bid}_fused.obj"), 
                        vertices = grid_pts.reshape(-1,3), 
                        scalars_for_color = z.squeeze(), colormap='jet',thres = sdf_thres)
                    
                if vis_belong:
                    draw_colored_points_to_obj(
                        os.path.join(save_dir, f"merge_pts_belong_{bid}.obj"), 
                        vertices = grid_pts.reshape(-1,3), 
                        scalars_for_color = belong_z.squeeze(), colormap='set')
                    
            ## number of nodes after merged larger than 1
            elif duplicate_box == 2:
                cnt = 0
                for m_order, new_sorted_ids  in zip(merge_order, sorted_fids):

                    cnt += 1
                    sdf_qs = []
                    for pid in new_sorted_ids:
                        z=[]
                        local_q = global_q
                        pnts_iterator = enumerate(torch.split(local_q, extract_batch_size, dim=0))

                        z = []
                        for i, pnts in pnts_iterator:
                            z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                            z.append(z_p)
                        z = torch.cat(z,dim=0)
                        z = z.numpy().astype(np.float32)

                        sdf_qs.append(z)
                        savefilename = os.path.join(save_dir, f"original_mc_surf_pts_{pid}_{bid}.obj")
                        marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)
                        
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
                    savefilename = os.path.join(save_dir, f"mc_surf_pts_{new_sorted_ids[kept_ids[0]]}_{bid}_{cnt}.obj") 
                    marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)

                    if vis_sdf:
                        draw_colored_points_to_obj(
                            os.path.join(save_dir, f"merge_grid_pts_{bid}_fused.obj"), 
                            vertices=grid_pts.reshape(-1,3), 
                            scalars_for_color=z.squeeze(), colormap='jet',thres = sdf_thres)
                        
                    if vis_belong:
                        draw_colored_points_to_obj(
                            os.path.join(save_dir, f"merge_pts_belong_{bid}.obj"), 
                            vertices=grid_pts.reshape(-1,3), 
                            scalars_for_color=belong_z.squeeze(), colormap='set')

                ## processing the unmerged patches as individual patches
                flatten_sorted_fids = [item for sublist in sorted_fids for item in sublist]
                left_ids = set(rbx.fids).difference(set(flatten_sorted_fids))
                for pid in left_ids:
                    local_q = global_q
                    pnts_iterator = enumerate(torch.split(local_q, extract_batch_size, dim=0))

                    z = []
                    for i, pnts in pnts_iterator:
                        z_p = (network(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(network.device).long()))[0].detach().cpu()
                        z.append(z_p)
                    z = torch.cat(z,dim=0)
                    z = z.numpy().astype(np.float32)

                    savefilename = os.path.join(save_dir, f"mc_surf_pts_{pid}_{bid}.obj")
                    marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)
                    savefilename = os.path.join(save_dir, f"original_mc_surf_pts_{pid}_{bid}.obj")
                    marching_cube(z.reshape(N, N, N).transpose([1, 0, 2]), scale, t, savefilename)
                    if vis_sdf:
                        draw_colored_points_to_obj(
                            os.path.join(save_dir, f"merge_grid_pts_{bid}_{pid}_fused.obj"), 
                            vertices=grid_pts.reshape(-1,3), 
                            scalars_for_color=z.squeeze(), colormap='jet',thres = sdf_thres)    
            else:
                raise NotImplementedError   
