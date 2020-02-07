'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
import pickle


class DatasetLoader(dataset):
    dataset_source_folder_path = None
    dataset_name = None
    load_type = 'Processed'

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(DatasetLoader, self).__init__(dName, dDescription)

    def load(self):
        if self.load_type == 'Raw':
            loaded_data = {'file_path': self.dataset_source_folder_path + 'data.txt'}
        elif self.load_type == 'Processed':
            f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
            loaded_data = pickle.load(f)
            f.close()
        return loaded_data
