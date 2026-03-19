import torch
import torch.nn.functional as F

def compute_patch_surface_loss(pred):
    surf_loss = pred.abs().mean()
    return surf_loss

def compute_patch_normal_loss(pred_norm,gt_norm):
    normals_loss = (pred_norm - gt_norm).norm(2, dim=1)
    normal_loss_value = normals_loss.mean()
    return normals_loss,normal_loss_value

def compute_patch_spatial_eiko_loss(off_surf_grads_nofuse):
    eikonal_loss = ((off_surf_grads_nofuse.norm(2, dim=-1) - 1) ** 2).mean()
    return eikonal_loss

def compute_patch_jitter_loss(pred_sdfs_jitter,gt_surf_jitter_sdfs):
    return F.l1_loss(pred_sdfs_jitter, gt_surf_jitter_sdfs)

def compute_patch_spatial_loss(spatial_pred):
    off_surf_penalty = torch.exp(-1e2 * torch.abs(spatial_pred)).mean()
    return off_surf_penalty

def compute_patch_loss_ss(loss_args, 
                       pred_sdfs,
                       pred_normals,
                       gt_surf_normals,
                       pred_normals_spatial,
                       pred_sdfs_jitter,
                       gt_surf_jitter_sdfs):
    
    surf_loss = compute_patch_surface_loss(pred_sdfs)
    _, norm_loss = compute_patch_normal_loss(pred_normals,gt_surf_normals)

    eikonal_loss = compute_patch_spatial_eiko_loss(pred_normals_spatial)
    jitter_sdf_loss = compute_patch_jitter_loss(pred_sdfs_jitter,gt_surf_jitter_sdfs)

    patch_loss = loss_args.surf * surf_loss + loss_args.surf_normal * norm_loss + loss_args.eikonal * eikonal_loss + loss_args.jitter_sdf * jitter_sdf_loss

    return {
        'patch_loss': patch_loss,
        'surf_loss': surf_loss,
        'norm_loss': norm_loss,
        'eikonal_loss': eikonal_loss,
        'jitter_sdf_loss': jitter_sdf_loss
    }
    
def compute_patch_loss(loss_args, 
                       pred_sdfs,
                       pred_normals,
                       gt_surf_normals,
                       pred_normals_spatial,
                       pred_sdfs_jitter,
                       gt_surf_jitter_sdfs):

    surf_loss = compute_patch_surface_loss(pred_sdfs)
    _, norm_loss = compute_patch_normal_loss(pred_normals,gt_surf_normals)

    eikonal_loss = compute_patch_spatial_eiko_loss(pred_normals_spatial)
    jitter_sdf_loss = compute_patch_jitter_loss(pred_sdfs_jitter,gt_surf_jitter_sdfs)

    patch_loss =  loss_args.surf * surf_loss + loss_args.surf_normal * norm_loss + loss_args.eikonal * eikonal_loss + loss_args.jitter_sdf * jitter_sdf_loss 

    return {
        'patch_loss':patch_loss,
        'surf_loss':surf_loss * loss_args.surf ,
        'norm_loss':norm_loss * loss_args.surf_normal,
        'eikonal_loss':eikonal_loss * loss_args.eikonal,
        'jitter_sdf_loss':jitter_sdf_loss * loss_args.jitter_sdf
    }