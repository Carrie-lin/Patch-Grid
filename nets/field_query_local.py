import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np 

from utils import replace_bdr_pts, filter_points_in_boxes
from preparation.lib.graph import define_merge_order


def re_organize_merge_grids(merge_grids, edge_types):

    merge_info = {}
    for bid, box in enumerate(merge_grids):

        face_ids_key = frozenset(box["face_ids"])
        assert len(box["face_ids"]) > 0

        box_sp = torch.Tensor(np.array(box['center']) - box['half_size']).reshape(1,3)
        box_length = torch.Tensor(np.array(box['half_size'] * 2)).reshape(1,1)
        
        ## for those merged cells
        if len(box["face_ids"]) > 1:

            if face_ids_key in merge_info.keys():

                merge_info[face_ids_key]['bids'].append(bid)
                merge_info[face_ids_key]['box_sps'] = torch.cat((merge_info[face_ids_key]['box_sps'], box_sp), dim=0)
                merge_info[face_ids_key]['box_lens'] = torch.cat((merge_info[face_ids_key]['box_lens'], box_length), dim=0)

            else:
                merge_order, sorted_fids, duplicate_box = define_merge_order(face_ids_key, edge_types)
                merge_info[face_ids_key] = {}
                merge_info[face_ids_key]['merge_order'] = merge_order
                merge_info[face_ids_key]['sorted_fids'] = sorted_fids
                merge_info[face_ids_key]['duplicate_box'] = duplicate_box
                merge_info[face_ids_key]['bids'] = [bid]
                merge_info[face_ids_key]['box_sps'] = box_sp
                merge_info[face_ids_key]['box_lens'] = box_length

        ## for those individual cells
        else:
            if face_ids_key in merge_info.keys():
                merge_info[face_ids_key]['bids'].append(bid)
                merge_info[face_ids_key]['box_sps'] = torch.cat((merge_info[face_ids_key]['box_sps'], box_sp), dim=0)
                merge_info[face_ids_key]['box_lens'] = torch.cat((merge_info[face_ids_key]['box_lens'], box_length), dim=0)
            else: 
                merge_info[face_ids_key] = {}
                merge_info[face_ids_key]['bids'] = [bid]
                merge_info[face_ids_key]['box_sps'] = box_sp
                merge_info[face_ids_key]['box_lens'] = box_length

        merge_info[face_ids_key]['fids'] = box["face_ids"]
        
    return merge_info


## query the field with only the input points and the local network
def return_query_field_local_func(local_net, merge_grids, edge_types, query_batch_size):

    local_net.eval()
    clustered_grids = re_organize_merge_grids(merge_grids, edge_types)

    def sdf_func(pts, local_net = local_net, merge_info = clustered_grids, query_batch_size = query_batch_size):
                
        ## initialize the sdf list
        sdf_list = np.ones((pts.shape[0], 1), dtype=np.float32) * 1000
        sdf_list = torch.Tensor(sdf_list)

        ## go over the merge order  
        for key_id in merge_info.keys():

            ## get the box list and the merge order 
            merge_item = merge_info[key_id]

            ## transform the numpy to be tensor
            box_arr = merge_item['box_sps']
            box_len = merge_item['box_lens']

            ## get the points in the box list
            filtered_pts, filtered_mask = filter_points_in_boxes(box_arr, box_len, pts, 0)
            #print('filtered_pts shape:', filtered_pts.shape)

            if filtered_pts.shape[0] <= 1:
                continue
            
            ## change the location of the points that located exactly on the right boundary of the box to avoid the error of the network 
            ## replace 1 with 0.999 
            filtered_pts = replace_bdr_pts(filtered_pts, box_arr, box_len)

            assert len(key_id) > 0

            ## go over the merge order 
            if len(key_id) == 1:

                ## single patch
                pid = merge_item['fids'][0]
                pnts_iterator = enumerate(torch.split(filtered_pts, query_batch_size, dim=0))

                z = []
                for i, pnts in pnts_iterator:
                    pnts = pnts.to(local_net.device)
                    z_p = (local_net(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(local_net.device).long()))[0].detach().cpu()
                    z.append(z_p)
                z = torch.cat(z,dim=0)
 
            else:
                ###########
                # Merging #
                ###########

                merge_order = merge_item['merge_order']
                sorted_fids = merge_item['sorted_fids']
                duplicate_box = merge_item['duplicate_box']
                fids = merge_item['fids']
                
                """
                return type:
                0: len(idx) = 0; in this case, input are some patches that are not connected in the space
                1: len(fids) > len(idx); in this case, some disconnected patches are missed while other connected merged.
                2: number of nodes after merged larger than 1; 
                    - in this case, there are more than 1 connected components in the grid (rarely seen)
                -1: all patches are merged into a single connected component
                """

                if duplicate_box == 0:

                    assert len(sorted_fids) == 0
                    z_list = []
                    for pid in fids:
                        local_q = filtered_pts
                        pnts_iterator = enumerate(torch.split(local_q, query_batch_size, dim=0))

                        z = []
                        for i, pnts in pnts_iterator:
                            pnts = pnts.to(local_net.device)
                            z_p = (local_net(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(local_net.device).long()))[0].detach().cpu()
                            z.append(z_p)
                        z = torch.cat(z,dim=0)
                        z_list.append(z)

                    ## get the z with the smallest absolute value
                    z = torch.stack(z_list, dim=0)
                    z_abs = z.abs()
                    z_min_index = torch.argmin(z_abs, dim = 0)
                    z = z[z_min_index, torch.arange(z.shape[1])]
                   
                elif duplicate_box == 1:

                    ## processing the merged patches
                    sdf_qs = []

                    for pid in sorted_fids:

                        pnts_iterator = enumerate(torch.split(filtered_pts, query_batch_size, dim=0))
                        
                        z = []
                        for i, pnts in pnts_iterator:
                            pnts = pnts.to(local_net.device)
                            z_p = (local_net(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(local_net.device).long()))[0].detach().cpu()
                            z.append(z_p)
                        z = torch.cat(z, dim = 0)
                        sdf_qs.append(z)

                    sdf_qs = torch.stack(sdf_qs, dim=0)
                    sdf_belong = torch.ones_like(sdf_qs) * -1
                    kept_ids = set(np.arange(len(sorted_fids)))

                    for how_to_merge in merge_order:
                        to_id = how_to_merge["edge"][0]
                        from_id = how_to_merge["edge"][1]
                        kept_ids.discard(from_id)
                        out = sdf_qs[[to_id, from_id],...]
                        if how_to_merge["property"] == 1:
                            out, out_index = torch.max(out, dim=0)
                        if how_to_merge["property"] == -1:
                            out, out_index = torch.min(out, dim=0)
                        sdf_qs[to_id] = out
                        sdf_belong[to_id] = out_index

                    z_list = []
                    for idx in kept_ids:
                        z = sdf_qs[idx]

                        z_list.append(z)

                        belong_z = sdf_belong[idx]
                        if belong_z.min() < 0 :
                            print("Wrong!")
                            exit(0)

                    #############################################################################
                    ## processing the unmerged patches as individual patches
                    left_ids = set(fids).difference(set(sorted_fids))
                    for pid in left_ids:
                        pnts_iterator = enumerate(torch.split(filtered_pts, query_batch_size, dim=0))

                        z = []
                        for i, pnts in pnts_iterator:
                            pnts = pnts.to(local_net.device)
                            z_p = (local_net(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(local_net.device).long()))[0].detach().cpu()
                            z.append(z_p)
                        z = torch.cat(z,dim=0)
                        z_list.append(z)

                    z = torch.stack(z_list, dim=0)
                    z_abs = z.abs()
                    z_min_index = torch.argmin(z_abs, dim = 0)
                    z = z[z_min_index, torch.arange(z.shape[1])]
                   
                ## -1: all patches are merged into a single connected component
                elif duplicate_box == -1:

                    sdf_qs = []
                    for pid in sorted_fids:
                        z=[]
                        pnts_iterator = enumerate(torch.split(filtered_pts, query_batch_size, dim=0))

                        z = []
                        for i, pnts in pnts_iterator:
                            pnts = pnts.to(local_net.device)
                            z_p = (local_net(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(local_net.device).long()))[0].detach().cpu()
                            z.append(z_p)
                        z = torch.cat(z,dim=0)
                        sdf_qs.append(z)

                    sdf_qs = torch.stack(sdf_qs, dim=0)  # [K, N, 1]
                    sdf_belong = torch.ones_like(sdf_qs) * -1
                    kept_ids = set(np.arange(len(sorted_fids)))

                    for how_to_merge in merge_order:
                        to_id = how_to_merge["edge"][0]
                        from_id = how_to_merge["edge"][1]
                        kept_ids.discard(from_id)
                        out = sdf_qs[[to_id, from_id],...]
                        if how_to_merge["property"] == 1:
                            out, out_index = torch.max(out, dim=0)
                        if how_to_merge["property"] == -1:
                            out, out_index = torch.min(out, dim=0)
                        sdf_qs[to_id] = out
                        sdf_belong[to_id] = out_index

                    kept_ids = list(kept_ids)
                    assert len(kept_ids) == 1
                    z = sdf_qs[to_id]
                    belong_z = sdf_belong[to_id]
                    if belong_z.min() < 0 :
                        print("Wrong!")
                        exit(0)

                elif duplicate_box == 2:

                    cnt = 0
                    z_list = []

                    for m_order, new_sorted_ids  in zip(merge_order, sorted_fids):
                        cnt += 1
                        sdf_qs = []
                        for pid in new_sorted_ids:
                            z=[]
                            pnts_iterator = enumerate(torch.split(filtered_pts, query_batch_size, dim=0))

                            z = []
                            for i, pnts in pnts_iterator:
                                pnts = pnts.to(local_net.device)
                                z_p = (local_net(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(local_net.device).long()))[0].detach().cpu()
                                z.append(z_p)
                            z = torch.cat(z,dim=0)
                            sdf_qs.append(z)

                        sdf_qs = torch.stack(sdf_qs, dim=0)  # [K, N, 1]
                        sdf_belong = torch.ones_like(sdf_qs) * -1
                        kept_ids = set(np.arange(len(new_sorted_ids)))

                        for how_to_merge in m_order:
                            to_id = how_to_merge["edge"][0]
                            from_id = how_to_merge["edge"][1]
                            kept_ids.discard(from_id)
                            out = sdf_qs[[to_id, from_id],...]
                            if how_to_merge["property"] == 1:
                                out, out_index = torch.max(out, dim=0)
                            if how_to_merge["property"] == -1:
                                out, out_index = torch.min(out, dim=0)
                            sdf_qs[to_id] = out
                            sdf_belong[to_id] = out_index

                        kept_ids = list(kept_ids)
                        assert len(kept_ids) == 1
                        z = sdf_qs[to_id]
                        belong_z = sdf_belong[to_id]
                        if belong_z.min() < 0 :
                            print("Wrong!")
                            exit(0)

                        z_list.append(z)

                    ## processing the unmerged patches as individual patches
                    flatten_sorted_fids = [item for sublist in sorted_fids for item in sublist]
                    
                    left_ids = set(fids).difference(set(flatten_sorted_fids))
                    for pid in left_ids:
                        pnts_iterator = enumerate(torch.split(filtered_pts, query_batch_size, dim=0))

                        z = []
                        for i, pnts in pnts_iterator:
                            pnts = pnts.to(local_net.device)
                            z_p = (local_net(pnts, pid = pid * (torch.ones((pnts.shape[0],1)).reshape(-1)).to(local_net.device).long()))[0].detach().cpu()
                            z.append(z_p)
                        z = torch.cat(z,dim=0)
                        z_list.append(z)

                    z = torch.stack(z_list, dim=0)
                    z_abs = z.abs()
                    z_min_index = torch.argmin(z_abs, dim = 0)
                    z = z[z_min_index, torch.arange(z.shape[1])]
                   
                else:
                    raise NotImplementedError   

            sdf_list[filtered_mask] = z.reshape(-1,1)

        return sdf_list
    
    return sdf_func

