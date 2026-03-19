from copy import deepcopy

class Node(object):
    def __init__(self, id):
        self.id = id
        self.neighbor_ids = []
        self.merged = False ## merges False --> not merged yet

        self.merged_node_ids = []
        self.merge_history = []

    ## get its neighbors
    def add_neighbor_ids(self, new_neigh_ids):
        for newid in new_neigh_ids:
            if newid == self.id:
                continue
            self.neighbor_ids.append(newid)
        self.neighbor_ids = list(set(self.neighbor_ids))

    # ## this is only used to finally update the node
    # def merge_node(self, node):
    #     print(f"Merging {node.id} to {self.id}")
    #     new_neighbor_ids = deepcopy(node.neighbor_ids)
    #     new_neighbor_ids.remove(self.id)
    #     self.add_neighbor_ids(new_neighbor_ids)
    #     # self.print()
    #     node.set_merged()

    def set_merged(self, merged=True):
        if merged == False:
            print("Warning: You are setting a merged node to unmerged condition! Are you sure?")
        self.merged = merged
        # print(f"set Node {self.id} merged")

    def empty_neighbors(self):
        self.neighbor_ids = None

    def print(self):
        print(f"Node id: {self.id} and Neighbors {self.neighbor_ids}; Merged nodes: {self.merged_node_ids}; Merged status: {self.merged}")
        print(f'Merge history: {self.merge_history}')


class Edge(object):
    def __init__(self, left, right, edge_property=1):
        ## these are nodes
        self.node_left = left
        self.node_right = right
        self.property = edge_property ## {-1,1} for concave and convex
        self.merged = False
    
    def __call__(self):
        return (self.node_left.id, self.node_right.id)

    def set_merged(self, merged=True):
        if merged == False:
            print("Warning: You are setting a merged node to unmerged condition! Are you sure?")
        self.merged = merged
        # print(f"set Edge {self.node_left.id}-{self.node_right.id} merged")

    def print(self):
        print(f"Edge {self.node_left.id}-{self.node_right.id} and property {self.property}; Merged: {self.merged}")




class MergingGraph(object):
    def __init__(self, nodes, id_edges, edge_property=None, verbose=0):
        self.verbose  = verbose
        self.merge_history = []

        #print('edge property:',edge_property) ## would be a list

        self.nodes = []
        ## create nodes
        self.nodes = [Node(nid) for nid in nodes]
        self.edges = []
        if edge_property is not None:
            for e, ep in zip(id_edges, edge_property):
                ## create the edges according to the nodes
                self.edges.append(Edge(self.nodes[e[0]], self.nodes[e[1]], ep))
                nid_left = e[0]
                nid_right = e[1]
                self.nodes[nid_left].add_neighbor_ids([nid_right])
                self.nodes[nid_right].add_neighbor_ids([nid_left])
        else:
            raise NotImplementedError
            # self.edges = [Edge(self.nodes[edge[0]], self.nodes[edge[1]]) for edge in edges]

        if self.verbose > 0:
            print("Graph built")

    def trace_merge_history(self, nid, history):
        node = self.nodes[nid]
        if len(node.merged_node_ids) == 0:
            return history
        else:
            for mid in node.merged_node_ids:
                self.trace_merge_history(mid, history)
                history.extend(self.nodes[mid].merge_history)
            return history
        

    def factorize_merge_order(self):
        if self.num_nodes() == 1:
            return self.merge_history
        else:
            # merge_history = []
            # for n in self.nodes:
            #     if not n.merged:
            #         merge_history.append(n.merge_history)
            # # return merge_history
            # print("original", merge_history)
            new_merge_history = []
            for n in self.nodes:
                if not n.merged:
                    m_merge_history = []
                    self.trace_merge_history(n.id, m_merge_history)
                    m_merge_history.extend(n.merge_history)
                    # print(m_merge_history)
                    new_merge_history.append(m_merge_history)
            # print("new", new_merge_history)
            return new_merge_history
    """
    ignore merged edges
    """
    def make_id_edges(self):
        id_edges = []
        for e in self.edges:
            if e.merged:
                continue
            id_edges.append(e())
        return id_edges


    def __get_edge(self, nid1, nid2):
        for e in self.edges:
            if e.merged: continue
            e1, e2 = e()
            if (e1 == nid1 and e2 == nid2) or (e2 == nid1 and e1 == nid2):
                return e
        return None

    def num_edges(self, show_merged=False):
        cnt = 0
        for e in self.edges:
            if e.merged and not show_merged: continue
            cnt += 1
        return cnt

    def num_nodes(self, show_merged=False):
        cnt = 0
        for n in self.nodes:
            if n.merged and not show_merged: continue
            cnt += 1
        return cnt

    def print(self, show_merged=False):
        print("===========")
        print("graph")
        self.print_edges(show_merged)
        self.print_nodes(show_merged)
        print("===========")

    def print_edges(self, show_merged=False):
        cnt = 0
        for e in self.edges:
            if e.merged and not show_merged: continue
            e.print() 
            cnt += 1
        print("num Edges:", cnt)


    def print_nodes(self, show_merged=False):
        cnt = 0
        for n in self.nodes:
            if n.merged and not show_merged: continue
            n.print()
            cnt += 1
        print("num Nodes:", cnt)

    ###############################################################

    def check_collapsable(self, node_1, node_2):
        set_1 = set(node_1.neighbor_ids)
        set_2 = set(node_2.neighbor_ids)
        inter_list = list(set_1.intersection(set_2))

        ## all of the intersection nodes need to be collapsable
        for inter in inter_list:
            edge_prop_1 = self.__get_edge(node_1.id, inter).property
            edge_prop_2 = self.__get_edge(node_2.id, inter).property
            if edge_prop_1*edge_prop_2 < 0:
                # print(f"Cannot merge {node_1.id} and {node_2.id}")
                return False
        return True

    def find_collapsable_edges(self,):
        for e in self.edges:
            if e.merged: continue
            flag = self.check_collapsable(e.node_left, e.node_right)
            if flag:
                return e
        return False


    def boardcast_merged_to_graph(self, edge):
        if self.verbose > 0:
            print("Boardcasting the merged information to graph members")
        # self.print(show_merged=True)

        kept_node = edge.node_left
        collapsed_node = edge.node_right
        kept_node.neighbor_ids.remove(collapsed_node.id)
        kept_node.merged_node_ids.append(collapsed_node.id)
        # for mn in collapsed_node.merged_node_ids:
        #     kept_node.merged_node_ids.append(mn)
        # for mh in collapsed_node.merge_history:
        #     kept_node.merge_history.append(mh)
        kept_node.merge_history.append({
                "edge": edge(),
                "property": edge.property,
            })

        if self.verbose > 0:
            print(f"Merging {collapsed_node.id} to {kept_node.id}")
        neighbor_ids_collasped_node = deepcopy(collapsed_node.neighbor_ids)
        collapsed_node.empty_neighbors()
        
        neighbor_ids_collasped_node.remove(kept_node.id)
        kept_node.add_neighbor_ids(neighbor_ids_collasped_node)
        collapsed_node.set_merged() ## set node 1 merged

        ## update the kept node
        edge.set_merged()

        ## processing all other incident edges
        nn_nodes = []
        for nnid in neighbor_ids_collasped_node:
            ee = self.__get_edge(collapsed_node.id, nnid)
            nn_node = self.nodes[nnid]
            nn_nodes.append(nn_node)
            ee.set_merged()

            ## if the to-be-added edge already exists, we assert they have the same property
            
            ee_exist = self.__get_edge(kept_node.id, nnid)
            if ee_exist is not None:
                assert ee_exist.property == ee.property
            else:
                # print("You are making use of Error; need to fix")
                ## assume nodes length does not change
                new_edge =Edge(kept_node, nn_node, ee.property)
                self.edges.append(new_edge)

        ## updating the neighboring node
        for nn_node in nn_nodes:
            nn_node.neighbor_ids.remove(collapsed_node.id)
            nn_node.add_neighbor_ids([kept_node.id])

        if self.verbose > 0:
            print("boardcast done")
            self.print(show_merged=True)
        
        self.merge_history.append(
            {
                "edge": edge(),
                "property": edge.property,
            })

    def merge(self):
        if self.verbose > 0:
            print("merging")

        ## self.edges should be updated on the fly
        cnt = 0
        flag = True
        while flag:
            edge = self.find_collapsable_edges()
            if edge is not False:
                cnt += 1
                self.boardcast_merged_to_graph(edge)
            else:
                flag = False

def define_merge_order(fids, edge_types):
    assert len(fids) > 1

    """
    return type:
    0: len(idx) = 0; in this case, input are some patches that are not connected in the space
    1: len(fids) > len(idx); in this case, some disconnected patches are missed while other connected merged.
    2: number of nodes after merged larger than 1; 
        - in this case, there are more than 1 connected components in the grid (rarely seen)
    -1: all patches are merged into a single connected component
    """

    # edge_types = self.network.edge_types
    edges = []
    edge_property = []

    ## build edge and its property
    for i in fids:
        for j in fids: 
            if i > j: continue
            if edge_types[int(i), int(j)] != 0:

                edges.append((i, j))
                edge_property.append(edge_types[int(i), int(j)])

    ## return vals
    merge_order = None
    box_duplicate = None
    idx = []

    ## sort fids with edges
    for eid in edges:
        idx.append(eid[0])
        idx.append(eid[1])
        # idx: [0, 4]
        # fids: [0, 4]

    ## not merge happens
    if len(idx) == 0:
        box_duplicate = 0
    ## merge happens cause there is an edge
    else:
        idx = list(set(idx)) ## unique idx since some patches might show up for twice
        nodes = [i for i in range(len(idx))] ## the index for the patches
        id_edges = []
        ## turn the patches represented edges to the index represented edges
        for e in edges:
            e0 = idx.index(e[0])
            e1 = idx.index(e[1])
            id_edges.append((e0, e1))
      
        graph_in_rbx = MergingGraph(nodes, id_edges, edge_property, verbose=0)
        graph_in_rbx.merge()
        merge_order = graph_in_rbx.factorize_merge_order()
        
        if graph_in_rbx.num_nodes() > 1:
            box_duplicate = 2
            new_merge_order = []
            new_idx = []

            #print('old merge order:', merge_order)
      
            for mh in merge_order:
                tmp_idx = set() ## non duplicated element
                for e in mh:
                    tmp_idx.add(idx[e["edge"][0]])
                    tmp_idx.add(idx[e["edge"][1]])
                ## tmp idx stores the exact patch id
                tmp_idx = list(tmp_idx) ## make a list for indexing
                tmp_mh = []
                for e in mh:
                    to_id = tmp_idx.index(idx[e["edge"][0]])
                    from_id = tmp_idx.index(idx[e["edge"][1]])
                    tmp_mh.append(
                        {
                            'edge': (to_id, from_id),
                            'property': e['property']
                        }
                    )
                new_merge_order.append(tmp_mh)
                new_idx.append(tmp_idx)

            #print('new merge order:', new_merge_order)
            return new_merge_order, new_idx, box_duplicate

        elif len(idx) < len(fids):
            box_duplicate = 1
        else:
            box_duplicate = -1

    return merge_order, idx, box_duplicate



if __name__ == "__main__":
    nodes = [0,1,2,3]
    id_edges = [(0,1), (1,2), (2,0), (2,3)]
    edge_property = [1, 1, -1, -1]

    nodes = [0,1,2,3]
    id_edges = [(0,1), (1,2), (2,0), (2,3), (0,3)]
    edge_property = [1, 1, -1, +1, -1]

    nodes = [0,1,2,3]
    id_edges = [(0,1), (1,2), (2,0), (2,3), (0,3), (1,3)]
    edge_property = [1, 1, -1, +1, -1, +1]


    nodes = [0,1]
    id_edges = [(0,1)]
    edge_property = [1]
    g = MergingGraph(nodes, id_edges, edge_property)

    print()
    g.merge()


    print()
    g.print(show_merged=True)
