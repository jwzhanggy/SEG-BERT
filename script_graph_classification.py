from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertGraphClassification import MethodGraphBertGraphClassification
from code.ResultSaving import ResultSaving
from code.Settings import Settings
import numpy as np
import torch


#---- IMDBBINARY, IMDBMULTI, MUTAG, NCI1, PTC, PROTEINS, COLLAB ----

seed = 0
dataset_name = 'MUTAG'
strategy = 'full_input'

np.random.seed(seed)
torch.manual_seed(seed)

if strategy == 'padding_pruning':
    if dataset_name in ['COLLAB', 'PROTEINS']:
        max_graph_size = 100
    elif dataset_name in ['MUTAG']:
        max_graph_size = 25
    else:
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


if dataset_name in ['IMDBBINARY', 'MUTAG', 'PROTEINS', 'NCI1']:
    nclass = 2
elif dataset_name in ['IMDBMULTI', 'PTC', 'COLLAB']:
    nclass = 3


#-----lr: MUTAG, IMDBBINARY
# 0.0001

#----lr: PTC
# 0.0005

#---- Fine-Tuning Task 1: Graph Bert Node Classification (Cora, Citeseer, and Pubmed) ----
if 1:
    for fold in range(1, 11):
        #---- hyper-parameters ----
        k = max_graph_size
        lr = 0.0005
        #---- max epochs, do an early stop when necessary ----
        max_epoch = 500
        ngraph = nfeature = max_graph_size
        x_size = nfeature
        hidden_size = intermediate_size = 32
        num_attention_heads = 2
        num_hidden_layers = 2
        y_size = nclass
        graph_size = ngraph
        residual_type = 'none'
        # --------------------------

        print('************ Start ************')
        print('GrapBert, dataset: ' + dataset_name + ', residual: ' + residual_type + ', k: ' + str(k) + ', hidden dimension: ' + str(hidden_size) +', hidden layer: ' + str(num_hidden_layers) + ', attention head: ' + str(num_attention_heads))
        # ---- objection initialization setction ---------------
        data_obj = DatasetLoader()
        data_obj.dataset_source_folder_path = './result/Padding/' + strategy + '/'
        data_obj.dataset_source_file_name = dataset_name
        data_obj.k = k

        bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
        method_obj = MethodGraphBertGraphClassification(bert_config)
        #---- set to false to run faster ----
        method_obj.spy_tag = True
        method_obj.max_epoch = max_epoch
        method_obj.lr = lr
        method_obj.fold = fold
        method_obj.strategy = strategy
        method_obj.load_pretrained_path = ''
        method_obj.save_pretrained_path =  ''

        result_obj = ResultSaving()
        result_obj.result_destination_folder_path = './result/AuGBert/'
        result_obj.result_destination_file_name = dataset_name + '_' + str(fold) + '_' + str(max_epoch) + '_' + residual_type + '_' + strategy

        setting_obj = Settings()

        evaluate_obj = None
        # ------------------------------------------------------

        # ---- running section ---------------------------------
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.load_run_save_evaluate()
        # ------------------------------------------------------


        #method_obj.save_pretrained('./result/PreTrained_GraphBert/' + dataset_name + '/node_classification_complete_model/')
        print('************ Finish ************')
#------------------------------------

