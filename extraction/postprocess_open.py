from glob import glob
import os
import trimesh
import torch

from extraction.postprocess import recompose_mesh_patches


def back_to_garment_check(net, sample_points, sdf_threshold = 0.00008):
    ## bdr_pids are all zeros
    bdr_pids = torch.zeros(sample_points.shape[0]).long().to(sample_points.device)
    pred_sdfs_bdr = net.forward_nograd(sample_points, bdr_pids)
    pred_sdfs_bdr_mean = (pred_sdfs_bdr.abs().mean())
    remain_flag = (pred_sdfs_bdr_mean < sdf_threshold)
    return remain_flag




def delete_boundary_patch(args, net, device):
    
    candidate_sub_mesh_list = []
    remain_sub_mesh_list = []

    ## load patches from sub_folder 
    sub_folder = os.path.join(args.mesh_path_prefix, f'{args.exp_name}_post')
    fnames = glob(os.path.join(sub_folder, "*.obj"))
    for fname in fnames:
        patch = trimesh.load(fname)
        candidate_sub_mesh_list.append(patch)

    test_sample_num = 10000
    ## sample points on each candidate patch and compute the sdf value
    for i in range(len(candidate_sub_mesh_list)):
        patch = candidate_sub_mesh_list[i]
        patch_samples = trimesh.sample.sample_surface(patch, test_sample_num)[0]
        patch_samples = torch.tensor(patch_samples).float().to(device)
        remain_flag = back_to_garment_check(net, patch_samples)
        if remain_flag:
            remain_sub_mesh_list.append(patch)
    
    final_sub_folder = os.path.join(args.mesh_path_prefix, f'{args.exp_name}_post_final')
    if not os.path.exists(final_sub_folder):
        os.makedirs(final_sub_folder)
    ## export all of the remain sub meshes
    for i in range(len(remain_sub_mesh_list)):
        remain_sub_mesh_list[i].export(f'{final_sub_folder}/composed_{i}.obj')
    
    ## recompose the patch
    recompose_mesh_patches(final_sub_folder, args.mesh_path_prefix, args.exp_name, cutted_flag = True)

    
    