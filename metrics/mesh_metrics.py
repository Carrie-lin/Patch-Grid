import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import trimesh
import numpy as np
import argparse
from utils import compute_pts_sdf

def compute_f_score(dis_arr2, dis_arr1, f_score_threshold):

    f_completeness = np.mean(dis_arr2 <= f_score_threshold)
    f_accuracy = np.mean(dis_arr1 <= f_score_threshold)
    f_score = 100 * 2 * f_completeness * f_accuracy / (f_completeness + f_accuracy)  # harmonic mean
    
    return f_score
        
def compute_chamfer(dis1, dis2):
    cd1 = dis1.mean()
    cd2 = dis2.mean()
    cd = (cd1 + cd2) / 2
    return cd

def compute_hausdorff(dis1, dis2):

    ## recon pts to gt mesh --> dis1
    ## gt pts to recon mesh --> dis2

    hd1 = dis1.max()
    hd2 = dis2.max()
        
    hd = max(hd1, hd2)
    return hd

def compute_normal_consistency(gt_mesh, recon_mesh, num_samples):

    gt_pts, gt_fid = trimesh.sample.sample_surface(gt_mesh, num_samples)
    recon_pts, recon_fid = trimesh.sample.sample_surface(recon_mesh, num_samples)

    _, _, gt_to_recon_fid = trimesh.proximity.closest_point(recon_mesh, gt_pts)
    _, _, recon_to_gt_fid = trimesh.proximity.closest_point(gt_mesh, recon_pts)

    gt_normals = gt_mesh.face_normals[gt_fid]
    gt_to_recon_normals = recon_mesh.face_normals[gt_to_recon_fid]

    recon_normals = recon_mesh.face_normals[recon_fid]
    recon_to_gt_normals = gt_mesh.face_normals[recon_to_gt_fid]

    cos1 = abs(np.matmul(gt_normals.reshape(-1,1,3), gt_to_recon_normals.reshape(-1,3,1))).mean()
    cos2 = abs(np.matmul(recon_normals.reshape(-1,1,3), -recon_to_gt_normals.reshape(-1,3,1))).mean()

    cos = (cos1 + cos2) / 2

    return cos

def compute_iou(gt_mesh, recon_mesh, query_batch_size):
    num_samples = 2**17
    samples = np.random.rand(num_samples, 3) * 2 - 1

    ## compute sdf of samples for gt mesh
    sdf_gt = compute_pts_sdf(gt_mesh, samples, query_batch_size)
    sdf_recon = compute_pts_sdf(recon_mesh, samples, query_batch_size)

    ## compute the intersection and union
    intersection = np.sum((sdf_gt < 0) & (sdf_recon < 0))
    union = np.sum((sdf_gt < 0) | (sdf_recon < 0))

    ## compute the iou
    iou = intersection / union

    return iou


if __name__ == '__main__':

    ## construct the parameters
    parser = argparse.ArgumentParser(description='Preprocess shapes.')
    parser.add_argument('--gt_path',type = str, default = None, required = True,  
                           help = 'The path of the ground truth mesh.')
    parser.add_argument('--recon_path',type = str, default = None, required = True,  
                           help = 'The path of the reconstructed mesh.')
    parser.add_argument('--sample_pts',type = int, default = 300000, 
                           help = 'The number of sample points for computing the metrics.')   
    parser.add_argument('--query_batch_size',type = int, default = 100000, 
                           help = 'The batchsize for evaluation.')   
    parser.add_argument('--f_score_threshold',type = float, default = 0.001,
                           help = 'The threshold for computing the f score.')
    parser.add_argument('--metrics_file', type = str, default = 'metrics.txt',
                           help = 'The file to save the metrics.')

    args = parser.parse_args()

    ## get the shape name from args.gt_path
    shape_name = os.path.split(args.gt_path)[1].split('.')[0]

    ## load the mesh
    gt_mesh = trimesh.load(args.gt_path, force='mesh')
    recon_mesh = trimesh.load(args.recon_path)

    ## sample points on both recon and gt mesh
    num_samples = args.sample_pts
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, num_samples)
    recon_pts, _ = trimesh.sample.sample_surface(recon_mesh, num_samples)

    ## compute points to surface distance
    dis1 = np.abs(compute_pts_sdf(gt_mesh, recon_pts, args.query_batch_size))
    dis2 = np.abs(compute_pts_sdf(recon_mesh, gt_pts, args.query_batch_size))

    ## compute hausdorff distance
    hd = compute_hausdorff(dis1, dis2)

    ## compute chamfer distance
    cd = compute_chamfer(dis1, dis2)
    
    ## compute f_score
    f_score = compute_f_score(dis1, dis2, args.f_score_threshold)

    nc = compute_normal_consistency(gt_mesh, recon_mesh, args.sample_pts)

    ## compute iou
    iou = compute_iou(gt_mesh, recon_mesh, args.query_batch_size)

    ## save the metrics
    with open(args.metrics_file, 'a+') as f:
        f.write(f'shape name: {shape_name} | ')
        f.write(f'cd: {cd} | ')
        f.write(f'hd: {hd} | ')
        f.write(f'iou: {iou} | ')
        f.write(f'nc: {nc} | ')
        f.write(f'f_score: {f_score}\n')
    
    print(f'{shape_name} | cd: {cd} | hd: {hd} | iou: {iou} | nc: {nc} | f_score: {f_score}')

