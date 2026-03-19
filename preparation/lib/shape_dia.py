import os 
import sys
from glob import glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import igl
import numpy as np
import trimesh 
import math
from utils import write_json

def compute_dia(patch_path, sample_num):

    v, f = igl.read_triangle_mesh(patch_path)

    mesh = trimesh.Trimesh(vertices=v, faces=f)
    samples, fids = trimesh.sample.sample_surface(mesh, sample_num)
    normals = mesh.face_normals[fids]

    ret = igl.shape_diameter_function(v, f, samples, -normals, len(samples))

    mean_v = ret.mean()
    if math.isnan(mean_v):
        mean_v = -1

    return mean_v


def compute_save_shape_dia(dataset_path, patch_num, sample_num):

    save_path = f'{dataset_path}/sdfs.json'
    dia_dict = {}
    for i in range(patch_num):
        patch_path = f'{dataset_path}/patches/{i}.obj'
        dia = compute_dia(patch_path,sample_num)
        dia_dict[str(i)] = dia 

    write_json(dia_dict, save_path)

    return dia_dict

if __name__ == "__main__":
    
    # read mesh
    v, f = igl.read_triangle_mesh(f'/home/guyinglin/document/research/tog/Patch-Grid-v1/data/train/00000002/39.obj')

    mesh = trimesh.Trimesh(vertices=v, faces=f)
    samples, fids = trimesh.sample.sample_surface(mesh, 1000)
    normals = mesh.face_normals[fids]

    ret = igl.shape_diameter_function(v, f, samples, -normals, len(samples))
    print(ret.mean())