import numpy as np
import torch

from code.DatasetLoader import DatasetLoader
from code.MethodWLNodeColoring import MethodWLNodeColoring
from code.MethodProcessRaw import MethodProcessRaw
from code.MethodPadding import MethodPadding
from code.ResultSaving import ResultSaving
from code.Settings import Settings

#---- IMDBBINARY, IMDBMULTI, MUTAG, NCI1, PTC,  PROTEINS, COLLAB ----

#----  REDDITBINARY, REDDITMULTI5K ----

seed = 0
dataset_name = 'PTC'
strategy = 'isolated_segment'

np.random.seed(seed)
torch.manual_seed(seed)

if strategy == 'padding_pruning':

    if dataset_name in ['COLLAB', 'PROTEINS']:
        max_graph_size = 100
    elif dataset_name in ['MUTAG']:
        max_graph_size = 25
    elif dataset_name in ['IMDBBINARY', 'IMDBMULTI', 'NCI1', 'PTC']:
        max_graph_size = 50

elif strategy == 'full_input':
    if dataset_name == 'IMDBBINARY':
        max_graph_size = 136
    elif dataset_name == 'IMDBMULTI':
        max_graph_size = 89
    elif dataset_name == 'MUTAG':
        max_graph_size = 28
    elif dataset_name == 'PTC':
        max_graph_size = 109

elif strategy == 'isolated_segment':
    if dataset_name == 'IMDBBINARY':
        max_graph_size = 140
    elif dataset_name == 'IMDBMULTI':
        max_graph_size = 100
    elif dataset_name == 'MUTAG':
        max_graph_size = 40
    elif dataset_name == 'PTC':
        max_graph_size = 120
    elif dataset_name == 'NCI1':
        max_graph_size = 120
    elif dataset_name == 'PROTEINS':
        max_graph_size = 620
    elif dataset_name == 'COLLAB':
        max_graph_size = 500

#---- Step 1: Load Raw Graphs for Train/Test Partition ----
if 1:
    print('************ Start ************')
    print('Preprocessing dataset: ' + dataset_name)
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.load_type = 'Raw'

    method_obj = MethodProcessRaw()

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/Preprocess/'
    result_obj.result_destination_file_name = dataset_name

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------

    print('************ Finish ************')
#------------------------------------


#---- Step 2: WL based graph coloring ----
if 1:
    print('************ Start ************')
    print('WL, dataset: ' + dataset_name)
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './result/Preprocess/'
    data_obj.dataset_source_file_name = dataset_name
    data_obj.load_type = 'Processed'

    method_obj = MethodWLNodeColoring()

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/WL/'
    result_obj.result_destination_file_name = dataset_name

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------

    print('************ Finish ************')
#------------------------------------

#---- Step 3: Graph Padding and Raw Feature/Tag Extraction ----
if 1:
    print('************ Start ************')
    print('WL, dataset: ' + dataset_name)
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './result/WL/'
    data_obj.dataset_source_file_name = dataset_name
    data_obj.load_type = 'Processed'

    method_obj = MethodPadding()
    method_obj.max_graph_size = max_graph_size

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/Padding/'
    result_obj.result_destination_file_name = dataset_name

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------

    print('************ Finish ************')
#------------------------------------
