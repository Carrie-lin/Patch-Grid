#from heapq import merge
import os
import sys
sys.path.append(os.getcwd())


## return the edge indicated by the prop
## prop is a nested set of propositions
## termination criterion is the input[2] is str type data
def distribute(prop, edges=None):
    if not isinstance(prop[2], str):
        p_left = [prop[0], prop[1], prop[2][1]]
        p_right = [prop[0], prop[1], prop[2][2]]
        prop_new = [prop[2][0], distribute(p_left, edges), distribute(p_right, edges)]
        return prop_new
    elif not isinstance(prop[1], str):
        p_left = [prop[0], prop[1][1], prop[2]]
        p_right = [prop[0], prop[1][2], prop[2]]
        prop_new = [prop[1][0], distribute(p_left, edges), distribute(p_right, edges)]
        return prop_new
    else:
        if edges is not None:
            duplicated = False
            for nd_e in edges:
                duplicated = ((prop[0] == nd_e[0]) and (prop[1] == nd_e[1]) and (prop[2] == nd_e[2])) \
                    or ((prop[0] == nd_e[0]) and (prop[2] == nd_e[1]) and (prop[1] == nd_e[2]))
                if duplicated: break
            if not duplicated:
                edges.append(prop)
        return prop


def get_prop_str_pos(prop, pos):
    if isinstance(prop[pos], str):
        return prop[pos]
    else:
        return get_prop_str_pos(prop[pos], pos)     


def convert_to_prop(merge_order, edges=None):
    temp_pool = []
    prop = []
    for m in merge_order:
        # print("a new merging edge", m)
        if m['property'] == 1:
            sign = 'max'
        elif m['property'] == -1:
            sign = 'min'
        else:
            raise NotImplementedError
        v0 = str(m['edge'][0])
        v1 = str(m['edge'][1])

        if len(prop) == 0:
            prop = [sign, v0, v1]
            prop = distribute(prop, edges) ## do the flattening
        else:
            prop_ret_1 = get_prop_str_pos(prop, 1)
            prop_ret_2 = get_prop_str_pos(prop, 2)
            if v0 == prop_ret_1 or v0 == prop_ret_2:
                added = False
                if len(temp_pool) != 0:
                    for tp in temp_pool:
                        if v1 == get_prop_str_pos(tp, 1) or v1 == get_prop_str_pos(tp, 2):
                            prop = [sign, prop, tp]
                            distribute(tp, edges)
                            temp_pool.remove(tp)
                            added = True
                            break ## break the temp_pool search
                if not added:
                    prop = [sign, prop, v1]
                prop = distribute(prop, edges) ## do the flattening
            elif v1 == prop_ret_1 or v1 == prop_ret_2:
                added = False
                if len(temp_pool) != 0:
                    for tp in temp_pool:
                        if v0 == get_prop_str_pos(tp, 1) or v0 == get_prop_str_pos(tp, 2):
                            prop = [sign, tp, prop]
                            distribute(tp, edges)
                            temp_pool.remove(tp)
                            added = True
                            break ## break the temp_pool search
                if not added:
                    prop = [sign, v0, prop]
                prop = distribute(prop, edges) ## do the flattening
            else:
                temp_pool.append([sign, v0, v1])


    if True:
        for tp in temp_pool:
            sign = tp[0]
            v0 = tp[1]
            v1 = tp[2]
            if len(prop) == 0:
                prop = [sign, v0, v1]
                prop = distribute(prop, edges) ## do the flattening
            else:
                prop_ret_1 = get_prop_str_pos(prop, 1)
                prop_ret_2 = get_prop_str_pos(prop, 2)
                #print(prop_ret_1, prop_ret_2)
                if v0 == prop_ret_1 or v0 == prop_ret_2:
                    added = False
                    if not added:
                        prop = [sign, prop, v1]
                    prop = distribute(prop, edges) ## do the flattening
                elif v1 == prop_ret_1 or v1 == prop_ret_2:
                    added = False
                    if not added:
                        prop = [sign, v0, prop]
                    prop = distribute(prop, edges) ## do the flattening
                else:
                    #print('Enter loop')
                    temp_pool.append([sign, v0, v1])
    else:
        for tp in temp_pool:
            sign = tp[0]
            v0 = tp[1]
            v1 = tp[2]
            prop = [sign, v0, v1]
            print('tp', tp)
            print("edges",edges)
            prop = distribute(prop, edges) ## do the flattening
            print("edges",edges)
            input()
    return prop

def convert_to_edge(prop, edges):
    if isinstance(prop[2], str):
        edges.append(prop)
    else:
        convert_to_edge(prop[1], edges)
        convert_to_edge(prop[2], edges)

'''
if __name__ == "__main__":
    
    # nodes = [0,1,2,3,4]
    # id_edges = [(0,1), (1,2), (2,0), (2,3), (1,4)]
    # edge_property = [1, 1, -1, -1, -1]
    
    # nodes = [0,1,2,3]
    # id_edges = [(0,1), (1,2), (2,0), (2,3)]
    # edge_property = [1, 1, -1, -1]

    nodes = [0,1,2,3,4,5,6,7]
    id_edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,0)]
    edge_property = [1, 1, 1, 1, -1, 1, -1, 1]

    # nodes = [0,1,2,3,4,5]
    # id_edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)]
    # edge_property = [1, 1, 1, 1, -1, 1]

    Amat = np.zeros((len(nodes), len(nodes)))
    for e in id_edges:
        Amat[e[0], e[1]] = 1.0
        Amat[e[1], e[0]] = 1.0
    print("Adjacent matrix\n", Amat) 

    g = MergingGraph(nodes, id_edges, edge_property)
    g.merge()
    g.print(show_merged=True)
    merge_order = g.factorize_merge_order()

    L = len(nodes)
    should_have=(L*L - L)/2
    have = len(merge_order)

    edges = []
    t0 = time.time()
    prop = convert_to_prop(merge_order, edges)
    t1 = time.time()

    print(t1-t0, L, len(edges)/should_have)
    if len(edges)/should_have > 1:
        print(merge_order)

    Amat_new = np.zeros((L, L))
    for e in edges:
        assert Amat_new[int(e[1]), int(e[2])] == 0
        assert int(e[1]) != int(e[2])
        Amat_new[int(e[1]), int(e[2])] = 1 if e[0] == 'max' else -1
        Amat_new[int(e[2]), int(e[1])] = 1 if e[0] == 'max' else -1
    print("Adjacent matrix\n", Amat_new) 

    # # ####################################################
    # # out_data = read_json('merge_order.json')
    
    # length_list = []
    # time_spent = []

    # for i, out in enumerate(out_data):
    #     # if i != 3:
    #     #     continue
    #     # print("case", i)
    #     if out is None:
    #         continue
    #     merge_order = out['merge_order']
    #     # print(merge_order)

    #     L = len(out['sorted_ids'])
    #     should_have=(L*L - L)/2
    #     have = len(merge_order)
    #     if have/should_have == 1:
    #         continue

    #     edges = []
    #     t0 = time.time()
    #     prop = convert_to_prop(merge_order, edges)
    #     t1 = time.time()

    #     # print(prop)

    #     time_spent.append(t1-t0)
    #     length_list.append(L)
    #     print(t1-t0, L, len(edges)/should_have)
    #     if len(edges)/should_have > 1:
    #         print(merge_order)

    #     Amat_new = np.zeros((L, L))
    #     for e in edges:
    #         assert Amat_new[int(e[1]), int(e[2])] == 0
    #         assert int(e[1]) != int(e[2])
    #         Amat_new[int(e[1]), int(e[2])] = 1 if e[0] == 'max' else -1
    #         Amat_new[int(e[2]), int(e[1])] = 1 if e[0] == 'max' else -1
    #     # print("Adjacent matrix\n", Amat_new) 

    # plt.scatter(length_list, time_spent)
    # plt.show()
'''