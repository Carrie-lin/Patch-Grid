import numpy as np
from scipy.spatial import cKDTree
import trimesh
import math
from utils import write_obj_file, process_grids_visualization, judge_if_intersect, mkdir_ifnotexists
from preparation.lib.octree.fv_octree import FVOctree

class BoundingBoxes():
    
    def __init__(self, patch_mesh, patch_id, resolution_power, extend_neighbor = False, points_flag = False):

        self.patch_mesh = patch_mesh
        self.patch_id = patch_id
        self.resolution_power = resolution_power
        self.extend_neighbor = extend_neighbor
        self.box_length = 2 / (2 ** resolution_power)

        self.surface_octree = FVOctree(
                                resolution_power,
                                np.array([-1,-1,-1]),
                                2,
                                patch_mesh,
                                draw_empty = False, points_flag = points_flag)
        
        self.surface_start_pts = np.array(self.surface_octree.box_list)  

        if extend_neighbor:
            ## get extended surface bounding boxes
            self.start_points = self.extend_surface_bounding_boxes()
        else:
            ## get surface bounding boxes
            self.start_points = self.surface_start_pts
        
    ## get the extended surface bounding boxes
    def extend_surface_bounding_boxes(self):
        ## for each axis in the self.surface_start_pts(this array is of shape [N,3], each line is a coordinate), we have 3 choices [0, -1, +1] * self.box_length, so we have 3^3 = 27 choices
        ## to get the newly generated coordinate (each line would generated 27 new coordinates)
        ## write it with numpy array

        # Create an array of shape [3, 3, 3] with values -1, 0, 1
        offsets = np.mgrid[-1:2, -1:2, -1:2].reshape(3, -1).T

        # Multiply offsets by box_length
        ## transform the offsets to be float
        offsets = offsets.astype(float)
        offsets *= self.box_length

        # Add offsets to each point in surface_start_pts
        new_coordinates = self.surface_start_pts[:, np.newaxis, :] + offsets

        # Now new_coordinates is an array of shape [N, 27, 3]
        new_coordinates = new_coordinates.reshape(-1, 3)

        ## remove the duplicate coordinates
        new_coordinates =  np.unique(new_coordinates, axis=0)

        threshold = 0.00001
        ## all of the coordinates are start points of the boxes, the length of the boxes is self.box_length, remove the points that are out of the boundary [-1,1] * [-1,1] * [-1,1]
        new_coordinates = new_coordinates[(new_coordinates >= -1 - threshold).all(axis=1) & (new_coordinates <= 1 - self.box_length + threshold).all(axis=1)]

        return new_coordinates

    ## save the bounding boxes
    def save_redboxes(self,data_prefix,extend_neighbor = False):

        mkdir_ifnotexists(f'{data_prefix}/bounds')
        if extend_neighbor:
            fname = f'{data_prefix}/bounds/box_p{self.patch_id}_r{int(2**self.resolution_power)}_extend.bin'
        else:
            fname = f'{data_prefix}/bounds/box_p{self.patch_id}_r{int(2**self.resolution_power)}_surface.bin'

        (self.start_points).tofile(fname)

    ## visualization for the bounding boxes
    def vis_grids(self):
        stepsize = 2.0 / (self.resolution)
        draw_vertices, draw_edge = process_grids_visualization(self.chosen_grid_start_points, stepsize)
        print(f'Draw octree visualization for bounding boxes to debug/test_octree_grids_{self.patch_id}.obj')
        write_obj_file(f'debug/test_octree_grids_{self.patch_id}.obj', V=draw_vertices, F=draw_edge,vid_start=0)


