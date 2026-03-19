import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np
import trimesh 

from option import parse_options
from utils import read_json, get_num_patches, sample_pts_in_merge_grids
from nets.OctreeSDF import OctreeSDF
from extraction.extractutils import get_predefined_redbox
from nets.field_query_local import return_query_field_local_func
from utils import compute_pts_sdf


def load_local_net(local_args, device):
    ## load adaptive resolution
    adaptive_path = f'{local_args.dataset_path}/adaptive_resolution.json'
    adaptive_patch_resolution = read_json(adaptive_path)
    adaptive_patch_resolution = {int(key): value for key, value in adaptive_patch_resolution.items()}

    num_patches = get_num_patches(f'{local_args.dataset_path}')

    net = OctreeSDF(local_args,
                    num_patches = num_patches,
                    device = device, 
                    adaptive_patch_resolution = adaptive_patch_resolution)

    net.load_state_dict(torch.load(local_args.local_path))
    net.to(device)
    net.eval()

    return net 


if __name__ == "__main__":
    
    ## prepare cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    ## load the local args
    local_parser = parse_options(return_parser=True)

    app_group = local_parser.add_argument_group('app')
    app_group.add_argument('--local_path',type = str, default = None,
                           help = 'Path to the local network.')
    app_group.add_argument('--field_metrics_file',type = str, default = None,
                           help = 'Path for the computed metrics.')
    app_group.add_argument('--sample_resolution',type = int, default = 2**8,
                           help = 'The resolution of the sample points.')
    app_group.add_argument('--scale_factor', type = float, default = 0.95,
                           help = 'Scale the sample points for field evaluation.')  
    app_group.add_argument('--query_batch_size', type = int, default = 50000,
                           help = 'Batch size of query.')

    local_args = local_parser.parse_args()

    ## load network
    local_net = load_local_net(local_args, device)

    ## load the edge type
    edge_type = np.loadtxt(os.path.join(local_args.dataset_path, 'connect_type.txt'))

    ## load the merge grids
    merge_path = f'{local_args.dataset_path}/redboxes_adapt.json'
    merge_grids = read_json(merge_path)

    ## load the gt mesh
    shape_name = local_args.dataset_path.split('/')[-1]
    gt_path = os.path.join(local_args.dataset_path, f'{shape_name}.obj')
    gt_mesh = trimesh.load(gt_path, process=False, maintain_order=True, validate=False)

    ## ===================== compute the error of local SDF ===================== 
    
    ## generate sample points according to merge grids
    merge_path = f'{local_args.dataset_path}/redboxes_adapt.json'
    merge_grids = read_json(merge_path)
    merge_struct = get_predefined_redbox(merge_path)
    merge_grid_pts = sample_pts_in_merge_grids(merge_struct, local_args.sample_resolution, local_args.scale_factor)
    merge_grid_pts = torch.cat(merge_grid_pts, dim = 0)

    ## compute field SDF
    local_sdf_func = return_query_field_local_func(local_net, merge_grids, edge_type, local_args.query_batch_size)
    local_sdf = local_sdf_func(merge_grid_pts)
    gt_local_sdf = compute_pts_sdf(gt_mesh, merge_grid_pts.numpy(), local_args.query_batch_size)
    local_sdf_err = np.abs(local_sdf.numpy().reshape(-1) - gt_local_sdf.reshape(-1))
    local_sdf_err_mean = local_sdf_err.mean()


    ## ===================== compute the error of sharp feature points ===================== 

    ## load the gt sharp points
    shape_name = local_args.dataset_path.split('/')[-1]
    feature_path = os.path.join(local_args.dataset_path, f'{shape_name}_curve.sample.xyz')
    feature_pts = np.loadtxt(feature_path)[:,:3]
    feature_pts = torch.from_numpy(feature_pts)

    ## put the sharp feature points into the field and compute the sdf error 
    sdf_err = local_sdf_func(feature_pts)
    sharp_feature_mean = np.abs(sdf_err).mean()
    
    ## write the errors into metrics file
    if local_args.field_metrics_file is not None:
        with open(local_args.field_metrics_file, 'a') as f:
            f.write(f'{shape_name} | field error: {local_sdf_err_mean} | sharp feature error: {sharp_feature_mean}')
    else:
        print(f'{shape_name} | field error: {local_sdf_err_mean} | sharp feature error: {sharp_feature_mean}',)

