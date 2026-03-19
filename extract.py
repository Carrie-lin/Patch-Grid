import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim.lr_scheduler
import numpy as np
import shutil
import trimesh
import open3d as o3d
import math

from utils import mkdir_ifnotexists, get_num_patches, read_json, filter_off_samples
from configs.option import parse_options
from nets.OctreeSDF import OctreeSDF
from extraction.extractutils import plot_rbx_marchingcube
from extraction.postprocess import cut_extended_meshes, recompose_mesh, back_to_field_check, recompose_mesh_patches, wrap_geometry, MeshInter, PatchBelong
from extraction.postprocess_open import delete_boundary_patch

def extract_surface(exp_name,
                    net,
                    save_prefix,
                    dataset_path,
                    extract_resolution,
                    device,
                    vis_sdf,
                    vis_belong,
                    append_layer_num,
                    extract_batch_size,
                    cut_boundary_length,
                    delete_info = None):
    
    my_path = os.path.join(save_prefix, exp_name)
    
    print(f'Save geometry to {my_path}.obj')
    
    mkdir_ifnotexists(my_path)

    edge_types = np.loadtxt(os.path.join(dataset_path, 'connect_type.txt'))
    
    plot_rbx_marchingcube(network = net, 
                          save_dir = my_path, 
                          extract_resolution = extract_resolution, 
                          dataset_path = dataset_path, 
                          device = device, 
                          edge_types = edge_types,
                          vis_sdf = vis_sdf,
                          vis_belong = vis_belong,
                          extract_batch_size = extract_batch_size,
                          append_layer_num = append_layer_num,
                          delete_patch_list = delete_info)

    ## post process for extendedly extracted meshes
    cut_extended_meshes(save_dir = my_path, 
                   geometric_prefix = save_prefix, 
                   exp_name = exp_name,
                   dataset_path = dataset_path,
                   cut_boundary_length = cut_boundary_length)

    recompose_mesh(save_dir = my_path, 
                   geometric_prefix = save_prefix, 
                   exp_name = exp_name)


    #geo_path = f'{my_path}_composed_all.obj' 
    #wrap_geometry(geo_path)
    #geo = trimesh.load(geo_path, process=True, maintain_order=True, validate=False)
    #geo.export(f'{my_path}_composed_all.obj')



def delete_wrong_patch(input_mesh_path, map_info, args, net, device):
    
    ## load the mesh with all the triagles and vertices position all the same
    mesh = trimesh.load(input_mesh_path, process=False, maintain_order=True, validate=False)

    ## get the number of triangles of mesh
    num_triangles = len(mesh.triangles)

    ## split the mesh into different components according to the map_info
    components = {}
    for i in range(num_triangles):
        patch_id = int(map_info[i])
        if patch_id not in components:
            components[patch_id] = []
        components[patch_id].append(i)
    
    sub_mesh_list = []
    sub_folder = os.path.join(args.mesh_path_prefix, f'{args.exp_name}_before_post')
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    else:
        shutil.rmtree(sub_folder)
        os.makedirs(sub_folder)

    ## save the mesh pieces
    for key in components:
        sub_mesh = mesh.submesh([components[key]], append=True)
        sub_mesh_list.append(sub_mesh)
        sub_mesh.export(f'{sub_folder}/{key}.obj')
    remain_sub_mesh_list = []

    ## collect the points and their corresponding submesh id
    all_pts = []
    sub_mesh_id = []
    for i in range(len(sub_mesh_list)):
        cur_vertices = sub_mesh_list[i].vertices
        all_pts.append(cur_vertices)
        sub_mesh_id.append(np.ones(len(sub_mesh_list[i].vertices)) * i)
    all_pts = np.concatenate(all_pts, axis = 0)
    sub_mesh_id = np.concatenate(sub_mesh_id, axis = 0)

    ## filter the all_pts that have one dimension larger than 1 or smaller than -1
    filter_flag, all_pts = filter_off_samples(all_pts, return_flag = True)
    sub_mesh_id = sub_mesh_id[filter_flag]

    remain_flag_list = back_to_field_check(all_pts,
                                        sub_mesh_id,
                                        args.open,
                                        net,
                                        args.mesh_path_prefix,
                                        args.dataset_path,
                                        args.extract_resolution,
                                        device,
                                        args.vis_sdf,
                                        args.vis_belong,
                                        args.append_layer_num,
                                        args.extract_batch_size,
                                        cut_boundary_length = args.remain_boundary,
                                        sdf_thres = args.sdf_thres)

    for i in range(len(remain_flag_list)):
        flag = remain_flag_list[i]
        if flag:
            remain_sub_mesh_list.append(sub_mesh_list[i])
    
    sub_folder = os.path.join(args.mesh_path_prefix, f'{args.exp_name}_post')
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    else:
        shutil.rmtree(sub_folder)
        os.makedirs(sub_folder)

    ## export all of the remain sub meshes
    for i in range(len(remain_sub_mesh_list)):
        remain_sub_mesh_list[i].export(f'{sub_folder}/composed_{i}.obj')
    
    ## recompose the patch
    recompose_mesh_patches(sub_folder, args.mesh_path_prefix, args.exp_name, cutted_flag = True)

def extract_individual_patches(exp_name,
                               save_prefix,
                               N_patches,
                               open_flag = False,
                               return_patchs = None):
    
    if return_patchs is None:
        pids = list(range(N_patches))
    else:
        pids = return_patchs

    my_path = os.path.join(save_prefix, exp_name)

    for pid in pids:
        recompose_mesh(save_dir = my_path, 
                   geometric_prefix = save_prefix, 
                   exp_name = exp_name,
                   patch_id = pid)
    
    if open_flag:
        ## merge all of the composed items to a single mesh
        recompose_mesh_patches(save_dir = my_path, 
                    geometric_prefix = save_prefix, 
                    exp_name = exp_name)


if __name__ == '__main__':

    parser = parse_options(return_parser=True)
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--mesh_path_prefix',type = str, required = True,
                           help = 'The path for the output mesh.')  
    app_group.add_argument('--per_patch', action='store_true',
                              help='To extract the patches seperately.')
    app_group.add_argument('--extract_resolution',type = int, default = 2**11,
                           help = 'The resolution of the extracted mesh.')
    app_group.add_argument('--vis_sdf', action = 'store_true',
                           help = 'To visualize the SDF per box.')
    app_group.add_argument('--vis_belong', action = 'store_true',
                           help = 'To visualize the belongings of each point.')
    app_group.add_argument('--return_patches', type = int, nargs='*', default = None,
                           help = 'The list of the id of the returned patches' )
    app_group.add_argument('--append_layer_num', type = int, default = 0,
                           help = 'To extend the marching cube in each box so that to avoid the holes.')
    app_group.add_argument('--sdf_thres', type = float, default = 0.000035,
                            help = 'The threshold for throwing away those unwanted patches.')
    app_group.add_argument('--remain_boundary', type = float, default = 0.0005,
                           help = 'Remain some boudary region to protect the surfaces that are too close to the box boundary.')
    app_group.add_argument('--extract_batch_size', type = int, default = 800000,
                           help = 'Batch size of query for the extraction.')

    args = parser.parse_args()

    # pick device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    ## load adaptive resolution
    adaptive_path = f'{args.dataset_path}/adaptive_resolution.json'
    adaptive_patch_resolution = read_json(adaptive_path)
    adaptive_patch_resolution = {int(key): value for key, value in adaptive_patch_resolution.items()}

    ## load delete patch list
    delete_path = f'{args.dataset_path}/delete_info.txt'
    if os.path.exists(delete_path):
        delete_info = np.loadtxt(delete_path)
        delete_info = delete_info.astype(np.int)
    else:
        delete_info = None

    ## get the number of patches
    if args.open:
        num_patches = len(adaptive_patch_resolution)
    else:
        num_patches = get_num_patches(f'{args.dataset_path}')

    ## load the network
    net = OctreeSDF(args,
                    num_patches = num_patches,
                    device = device, 
                    adaptive_patch_resolution = adaptive_patch_resolution)
    
    net.load_state_dict(torch.load(args.pretrained))
    net.to(device)
    net.eval()
    
    print("Total number of parameters: {}".format(sum(p.numel() for p in net.parameters())))

    ## extract the whole shape
    extract_surface(args.exp_name, 
                    net,
                    args.mesh_path_prefix,
                    args.dataset_path,
                    args.extract_resolution,
                    device,
                    args.vis_sdf,
                    args.vis_belong,
                    args.append_layer_num,
                    args.extract_batch_size,
                    cut_boundary_length = args.remain_boundary,
                    delete_info = delete_info)
    
    ## extract the geometry of the individual patches
    extract_individual_patches(args.exp_name,
                            args.mesh_path_prefix,
                            num_patches,
                            return_patchs = args.return_patches)


