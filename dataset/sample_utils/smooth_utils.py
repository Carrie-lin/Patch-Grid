import torch 
import trimesh
from utils import return_box_cutted_mesh

def sample_on_mesh(mesh, sample_num):

    points, fids = trimesh.sample.sample_surface(mesh, sample_num)
    points = torch.from_numpy(points)
    normals = mesh.face_normals[fids]
    normals = torch.from_numpy(normals)
    fake_mask = torch.ones((points.shape[0],1)) * (-1) 
    fake_mask = fake_mask.reshape(-1)

    return points,normals,fake_mask


def select_smooth_box(ready_to_use_smooth_type):
    
    box_list = []
    box_smooth_list = {}
    for pid1 in ready_to_use_smooth_type.keys():
        for pid2 in ready_to_use_smooth_type[pid1].keys():
            box_list.extend(ready_to_use_smooth_type[pid1][pid2])

    box_list = list(set(box_list))

    for bid in box_list:
        for pid1 in ready_to_use_smooth_type.keys():
            for pid2 in ready_to_use_smooth_type[pid1].keys():
                if bid in ready_to_use_smooth_type[pid1][pid2]:
                    if bid in box_smooth_list.keys():
                        box_smooth_list[bid].append(pid1)
                        box_smooth_list[bid].append(pid2)
                    else:
                        box_smooth_list[bid] = [pid1, pid2]
    
    for bid in box_smooth_list.keys():
        cur_list = box_smooth_list[bid]
        cur_list = list(set(cur_list))
        box_smooth_list[bid] = cur_list

    return box_list, box_smooth_list

def process_smooth_addi_refer_gt(num_patches, smooth_extend_type, box_info_surface, patch_meshes, adaptive_patch_grids, args):
    
    add_pts_list = [-1 for i in range(num_patches)]
    add_norm_list = [-1 for i in range(num_patches)]
    #add_redbox_id_list = [-1 for i in range(num_patches)]
    add_pid_list = [-1 for i in range(num_patches)]

    for i in range(num_patches):

        ## go over the patches, if the patch has smooth connections
        if str(i) in smooth_extend_type.keys():
            
            box_list = smooth_extend_type[str(i)].keys()
            box_list = [int(box_item) for box_item in box_list]

            connected_patch_id_list = []
            for bid in box_list:
                cpid = smooth_extend_type[str(i)][str(bid)].keys()
                cpid = [int(cpid_item) for cpid_item in cpid]
                connected_patch_id_list.append(cpid)

            box_start_p_list = box_info_surface[i][box_list]

            cur_patch_reso = 2**adaptive_patch_grids[i]
            box_len = 2 / cur_patch_reso

            connected_patch_pts_items = []
            connected_patch_nrms_items = []
            #connected_patch_redbox_id_items = []
            
            ## go over every feature volume box
            for bid in range(box_start_p_list.shape[0]):

                start_p = box_start_p_list[bid]
                ci_list = connected_patch_id_list[bid]

                cur_mesh = patch_meshes[i]

                ## cutted current mesh
                cutted_cur_mesh = return_box_cutted_mesh(cur_mesh, start_p, box_len)

                cutted_connected_mesh_list = []
                for ci in ci_list:
                    connected_mesh = patch_meshes[ci]
                    cutted_connected_mesh = return_box_cutted_mesh(connected_mesh,start_p,box_len)
                    cutted_connected_mesh_list.append(cutted_connected_mesh)

                ## area distribution 
                area_cur = cutted_cur_mesh.area
                area_connected_sum = 0
                area_connected_list = []
                
                for ci in range(len(ci_list)):
                    area_connected = cutted_connected_mesh_list[ci].area 
                    area_connected_sum += area_connected
                    area_connected_list.append(area_connected)

                ## sample individually
                sample_num_cur = int(args.box_surf_num * area_cur / (area_cur + area_connected_sum))
                sample_num_connected_list = []
                
                for ci in range(len(ci_list)):
                    area_connected = area_connected_list[ci]
                    sample_num_connected = int(args.box_surf_num * area_connected / (area_cur + area_connected_sum))
                    sample_num_connected_list.append(sample_num_connected)
                
                points_cur, normals_cur, _ = sample_on_mesh(cutted_cur_mesh, sample_num_cur)
                connected_patch_pts_items.append(points_cur)
                connected_patch_nrms_items.append(normals_cur)
                #connected_patch_redbox_id_items.append(fake_mask_cur)

                for ci in range(len(ci_list)):
                    cutted_connected_mesh = cutted_connected_mesh_list[ci]
                    sample_num_connected = sample_num_connected_list[ci]
                    points_connected, normals_connected, _ = sample_on_mesh(cutted_connected_mesh,sample_num_connected)

                    connected_patch_pts_items.append(points_connected)
                    connected_patch_nrms_items.append(normals_connected)
                    #connected_patch_redbox_id_items.append(fake_redbox_id_connected)

            add_pts_list[i] = torch.cat(connected_patch_pts_items)
            add_norm_list[i] = torch.cat(connected_patch_nrms_items)
            add_pid = (torch.ones((add_pts_list[i].shape[0],1)) * i)
            add_pid_list[i] = add_pid
    


    return {
        'smooth_points': add_pts_list,
        'smooth_normals': add_norm_list,
        'smooth_pid': add_pid_list
    }

