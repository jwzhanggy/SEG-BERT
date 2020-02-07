'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import torch
import networkx as nx
import numpy as np
from sklearn.model_selection import StratifiedKFold
from code.base_class.method import method

class MethodPadding(method):
    seed = None
    data = None
    max_graph_size = None
    label_dict = {}

    def padding(self, graph_dict, max_size):
        node_tags = [max_size+1]*max_size
        node_degrees = [0] * max_size
        wl_tags = [0]*max_size
        w_list = []

        graph = graph_dict['graph']
        if graph_dict['label'] not in self.label_dict:
            self.label_dict[graph_dict['label']] = len(self.label_dict)
        y = self.label_dict[graph_dict['label']]
        wl_code = graph_dict['node_WL_code']

        node_list = list(graph.nodes)
        idx_map = {j: i for i, j in enumerate(node_list)}
        for i in range(max_size):
            w = [0.0] * max_size
            if i < len(node_list):
                node = node_list[i]
                node_tags[i] = node
                node_degrees[i] = graph.degree(node)
                wl_tags[i] = wl_code[i]
                neighbor_list = list(graph.neighbors(node))
                for neighbor in neighbor_list:
                    if idx_map[neighbor] >= max_size: continue
                    w[idx_map[neighbor]] = 1.0
            w_list.append(w)

        return node_tags, node_degrees, wl_tags, w_list, y

    def run(self):
        processed_graph_data = []
        max_graph_size = self.max_graph_size
        for i in range(len(self.data['graph_list'])):
            graph = self.data['graph_list'][i]
            tag, degree, wl, w, y = self.padding(graph, max_graph_size)
            processed_graph_data.append({'id': i, 'tag': tag, 'degree': degree, 'weight': w, 'wl_tag': wl, 'y': y})
        self.data['processed_graph_data'] = processed_graph_data
        return self.data