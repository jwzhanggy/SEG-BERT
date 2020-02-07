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

class MethodProcessRaw(method):
    seed = None
    data = None

    def load_raw_graph_list(self, file_path):
        g_list = []
        label_dict = {}
        feat_dict = {}

        graph_size_list = []
        with open(file_path, 'r') as f:
            n_g = int(f.readline().strip())
            for i in range(n_g):
                row = f.readline().strip().split()
                graph_size_list.append(int(row[0]))
                n, l = [int(w) for w in row]
                if not l in label_dict:
                    mapped = len(label_dict)
                    label_dict[l] = mapped
                g = nx.Graph()
                n_edges = 0
                for j in range(n):
                    g.add_node(j)
                    row = f.readline().strip().split()
                    row = [int(w) for w in row]
                    n_edges += row[1]
                    for k in range(2, len(row)):
                        g.add_edge(j, row[k])

                assert len(g) == n
                g_list.append({'graph': g, 'label': l})

        print('# classes: %d' % len(label_dict), '; # data: %d' % len(g_list), '; max graph size: %d' % max(graph_size_list))
        return g_list, graph_size_list

    def separate_data(self, graph_list, seed):
        train_idx_dict = {}
        test_idex_dict = {}

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        labels = [graph['label'] for graph in graph_list]
        fold_count = 1
        for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
            train_idx_dict[fold_count] = train_idx
            test_idex_dict[fold_count] = test_idx
            fold_count += 1

        return train_idx_dict, test_idex_dict

    def run(self):

        file_path = self.data['file_path']
        graph_list, graph_size_list = self.load_raw_graph_list(file_path)
        train_idx_dic, test_idx_dict = self.separate_data(graph_list, self.seed)
        max_size = max(graph_size_list)
        return {'graph_list': graph_list, 'max_graph_size': max_size, 'graph_size_list': graph_size_list, 'train_idx': train_idx_dic, 'test_idx': test_idx_dict}
