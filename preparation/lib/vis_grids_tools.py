import numpy as np
import re
from utils import read_json, write_obj_file, process_grids_visualization

## draw grids
def draw_grids(chosen_grid_start_points, stepsize, file_path, octree = False):

    draw_vertices, draw_edge = process_grids_visualization(chosen_grid_start_points, stepsize, octree = octree)
    print(f'draw box visualization to {file_path}')

    write_obj_file(file_path, V = draw_vertices, F = draw_edge, vid_start = 0)

## visualization of merge grid
def draw_mg(args):

    redboxes = read_json(args.input_file)

    chosen_grid = []
    half_size = []
    bid_count = 0

    ## select the merge grids that contain the specified patch list
    if args.merge_pid:
        bid = 0
        for rbx in redboxes:
            cur_patch_list = rbx['face_ids']
            ## judge if every element in merge_pid is in cur_patch_list
            if len(cur_patch_list) == 2:
                bid += 1
                continue
            if all([pid in cur_patch_list for pid in args.merge_pid]):
                chosen_grid.append(rbx['center'])
                half_size.append(rbx['half_size'])
                print(f'{bid} info:',rbx)
            bid += 1

    else:
        for rbx in redboxes:  
            
            if args.bid is None:  
                chosen_grid.append(rbx['center'])
                half_size.append(rbx['half_size'])
            else:
                if (args.bid) == bid_count:
                    chosen_grid.append(rbx['center'])
                    half_size.append(rbx['half_size'])
            bid_count += 1
            


    chosen_grid = np.array(chosen_grid)
    half_size = np.array(half_size).reshape(-1,1)
    stepsize = half_size * 2
    chosen_grid_start_points = chosen_grid - half_size
    draw_grids(chosen_grid_start_points, stepsize, args.output_file, octree = True)

## visualization of feature volume
def draw_fv(args):
    
    shape_path = args.input_file.split('bounds/')[0]
    file_name = args.input_file.split('bounds/')[1]
    adapt_reso_path = f'{shape_path}adaptive_resolution.json'
    adaptive_resolution_dict = read_json(adapt_reso_path)

    match = re.search('p(\d+)', file_name)
    patch_id = int(match.group(1))

    resolution_power = adaptive_resolution_dict[str(patch_id)]
    resolution = int(2 ** resolution_power)
    stepsize = 2.0 / resolution
    box_start_pts = np.fromfile(args.input_file).reshape(-1, 3)
    
    if not(args.fbid is None):
        box_start_pts = box_start_pts[args.fbid].reshape(-1,3)
    
    draw_grids(box_start_pts, stepsize, args.output_file, octree = False)