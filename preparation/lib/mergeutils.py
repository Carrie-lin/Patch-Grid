import os, sys
sys.path.append(os.getcwd())
import numpy as np
from scipy.spatial import cKDTree
from utils import write_json, build_grids
from preparation.lib.graph import define_merge_order
from preparation.lib.distribute_minmax import convert_to_prop

class MergeBaker():

    def __init__(self, redboxes, input_folder): 

        self.redboxes = redboxes
        edge_types = np.loadtxt(os.path.join(input_folder, 'connect_type.txt'))
        #edge_types[edge_types == 2] = 1
        groups = redbox_grouping(redboxes)
        self.merge_order = bake_merging_order_group(groups, edge_types)

        ## add adj
        for i, out in enumerate(self.merge_order):
            if out is None:
                continue
            mh = out['merge_order']
            if mh is None:
                continue
            L = len(out['sorted_ids'])
            should_have = (L*L - L)/2
            have = len(mh)

            edges = []
            prop = fake_convert_to_prop(mh, edges)
            #prop = convert_to_prop(mh, edges)

            Amat = np.zeros((L, L))
            for e in edges:
                assert Amat[int(e[1]), int(e[2])] == 0
                assert int(e[1]) != int(e[2])
                Amat[int(e[1]), int(e[2])] = 1 if e[0] == 'max' else -1
                Amat[int(e[2]), int(e[1])] = 1 if e[0] == 'max' else -1
            out['Adj'] = list(Amat.flatten())
            
        self.process_out()
        
        write_json(self.merge_order, os.path.join(input_folder, f"merge_order_adapt.json"))

    def process_out(self):
        for merge_item in self.merge_order:
            if merge_item:
                new_sort = []
                new_patch = []
                for id in merge_item['sorted_ids']:
                    new_sort.append(int(id))
                for id in merge_item['patch_ids']:
                    new_patch.append(int(id))
                merge_item['sorted_ids'] = new_sort
                merge_item['patch_ids'] = new_patch


def fake_convert_to_prop(merge_order, edges):

    prop = []
    for m in merge_order:
        if m['property'] == 1:
            sign = 'max'
        elif m['property'] == -1:
            sign = 'min'
        else:
            raise NotImplementedError
        v0 = str(m['edge'][0])
        v1 = str(m['edge'][1])

        prop.append([sign, v0, v1])
    #print('prop:',prop) # prop: [['max', '0', '1']]
    #exit(0)
    edges.extend(prop)

    return prop


### this function returns a mask indicating which points are inside which red box
# mnfld_pnts_of_patches: [B, N, 3] (numpy.array)
# rebbox_list: a list of L elements
def scatter_points_to_redboxes(points, rebbox_list):
    grid_pts = []
    for rbx in rebbox_list:
        grid_pts.append(rbx.center)
    grid_tree = cKDTree(np.array(grid_pts))
    _, inside_mask = grid_tree.query(points, k=1, p=1, workers=4)
    return inside_mask

def build_grid_tree(rebbox_list):
    grid_pts = []
    for rbx in rebbox_list:
        grid_pts.append(rbx.center)
    grid_tree = cKDTree(np.array(grid_pts))
    return grid_tree

### this function returns a merging order for each red box
def bake_merging_order(rebbox_list, edge_types):
    rbx_merge_order_list = []
    for bid, rbx in enumerate(rebbox_list):
        if len(rbx.fids) > 1:
            merge_order, sorted_idx, box_duplicate = define_merge_order(rbx.fids, edge_types)
            if box_duplicate == 2:
                for mh, mh_idx in zip(merge_order, sorted_idx):
                    merge_info = {
                        "merge_order": mh,
                        "sorted_ids": mh_idx,
                        "duplicate": 2,
                        "patch_ids": rbx.fids,
                        "box_id": bid
                    }
                    rbx_merge_order_list.append(merge_info) ## len(rfx.fids) > 1
            else:
                merge_info = {
                    "merge_order": merge_order,
                    "sorted_ids": sorted_idx,
                    "duplicate": box_duplicate,
                    "patch_ids": rbx.fids,
                    "box_id": bid
                }
                rbx_merge_order_list.append(merge_info) ## len(rfx.fids) > 1
        else:
            assert len(rbx.fids) == 1
            rbx_merge_order_list.append(None)
    
    return rbx_merge_order_list


def bake_merging_order_group(grouped_redboxes, edge_types):
    rbx_merge_order_list = []
    ## go over each merge group
    for rbx in grouped_redboxes: ## rbx may contain a lot of boxes
        if len(rbx['fids']) > 1:
            merge_order, sorted_idx, box_duplicate = define_merge_order(rbx['fids'], edge_types)
            if box_duplicate == 2:
                for mh, mh_idx in zip(merge_order, sorted_idx):
                    merge_info = {
                        "merge_order": mh,
                        "sorted_ids": mh_idx,
                        "duplicate": 2,
                        "patch_ids": rbx['fids'],
                        "box_id": rbx['bids']
                    }
                    rbx_merge_order_list.append(merge_info) ## len(rfx.fids) > 1
            else:
                merge_info = {
                    "merge_order": merge_order,
                    "sorted_ids": sorted_idx,
                    "duplicate": box_duplicate,
                    "patch_ids": rbx['fids'],
                    "box_id": rbx['bids']
                }
                rbx_merge_order_list.append(merge_info) ## len(rfx.fids) > 1
        else:
            ## if there is only one patch inside the box, we don't need any merge item.
            ## ignore the single patch case
            assert len(rbx['fids']) == 1
            rbx_merge_order_list.append(None)

    return rbx_merge_order_list

def redbox_grouping(redbox_list):

    group_list = []
    for bid, rbx in enumerate(redbox_list):
        added = False
        for g in group_list:
            ## if the patches are the same, we group the boxes
            if set(g['fids']) == set(rbx.fids):
                g['bids'].append(bid)
                added = True
                break
        if not added:
            group_list.append(
                {
                    "fids": rbx.fids,
                    "bids": [bid]
                }
            )

    return group_list

class RedBox():
    def __init__(self, center, half_size):
        self.center = center
        self.half_size = half_size
        self.local_offset= None
        self.height = None

    def build_box(self, N, custom_half_size = None):
        if custom_half_size:
            self.local_offset = build_grids(N).reshape(-1,3) * custom_half_size
        else:
            self.local_offset = build_grids(N).reshape(-1,3) * self.half_size

    def get_grid_pts(self):
        return self.center[None,:] + self.local_offset

    def set_face_ids(self, fids):
        if type(fids) is list:
            self.fids = fids
        else:
            self.fids = list(fids)

    def check_inside(self, q, epsilon=0.0):
        lows = self.center - np.array([self.half_size, self.half_size, self.half_size]) - epsilon
        hghs = self.center + np.array([self.half_size, self.half_size, self.half_size]) + epsilon
        lows_t = np.sum(q >= lows, axis=-1) == 3
        hghs_t = np.sum(q <= hghs, axis=-1) == 3
        return (np.bitwise_and(lows_t, hghs_t)).any()
    
    def set_height(self,height):
        self.height = height

    def get_box_bounds(self, epsilon=0.0):
        low_coord = self.center - np.array([self.half_size, self.half_size, self.half_size]) - epsilon
        high_coord = self.center + np.array([self.half_size, self.half_size, self.half_size]) + epsilon
        return low_coord, high_coord

    def write_out(self):
        content = {
            "center": list(self.center),
            "face_ids": self.fids,
            "half_size": self.half_size,
            "height":self.height
        }
        return content

