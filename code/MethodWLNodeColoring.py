'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.method import method
import hashlib


class MethodWLNodeColoring(method):
    data = None
    max_iter = 2

    def setting_init(self, node_list, link_list):
        node_color_dict = {}
        node_neighbor_dict = {}
        for node in node_list:
            node_color_dict[node] = 1
            node_neighbor_dict[node] = {}

        for pair in link_list:
            u1, u2 = pair
            if u1 not in node_neighbor_dict:
                node_neighbor_dict[u1] = {}
            if u2 not in node_neighbor_dict:
                node_neighbor_dict[u2] = {}
            node_neighbor_dict[u1][u2] = 1
            node_neighbor_dict[u2][u1] = 1
        return node_color_dict, node_neighbor_dict

    def WL_recursion(self, node_list, node_color_dict, node_neighbor_dict):
        iteration_count = 1
        while True:
            new_color_dict = {}
            for node in node_list:
                neighbors = node_neighbor_dict[node]
                neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
                color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()
                new_color_dict[node] = hashing
            color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]
            if node_color_dict == new_color_dict or iteration_count == self.max_iter:
                return node_color_dict
            else:
                node_color_dict = new_color_dict
            iteration_count += 1

    def run(self):
        print('computing WL code of graph nodes...')
        for graph_index in range(len(self.data['graph_list'])):
            graph = self.data['graph_list'][graph_index]
            node_list = graph['graph'].nodes
            link_list = graph['graph'].edges
            node_color_dict, node_neighbor_dict = self.setting_init(node_list, link_list)
            node_color_dict = self.WL_recursion(node_list, node_color_dict, node_neighbor_dict)
            node_color_list = []
            for node in node_list:
                node_color_list.append(node_color_dict[node])
            graph['node_WL_code'] = node_color_list
            self.data['graph_list'][graph_index] = graph
        return self.data
