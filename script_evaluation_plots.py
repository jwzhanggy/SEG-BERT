import numpy as np
import matplotlib.pyplot as plt
from code.ResultSaving import ResultSaving

#---------- clustering results evaluation -----------------


#---- IMDBBINARY, IMDBMULTI, MUTAG, NCI1, PTC,  PROTEINS,   COLLAB, REDDITBINARY, REDDITMULTI5K ----

#---- isolated_segment, padding_pruning, full_input
strategy = 'isolated_segment'
dataset_name = 'IMDBMULTI'
residual_type = 'none'

if 1:
    epoch_number = 500
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/AuGBert/' + strategy + '/' + dataset_name + '/'

    result_list = []
    time_list = []
    for fold in range(1, 11):
        result_obj.result_destination_file_name = dataset_name + '_' + str(fold) + '_' + str(epoch_number) + '_' + residual_type + '_' + strategy
        loaded_result = result_obj.load()
        time_list.append(sum([loaded_result[epoch]['time'] for epoch in loaded_result]))
        result_list.append(np.max([loaded_result[epoch]['acc_test'] for epoch in loaded_result]))
    print('accuracy: {:.2f}$\pm${:.2f}'.format(100*np.mean(result_list), 100*np.std(result_list)))
    print('time: {:.2f}$\pm${:.2f}'.format(np.mean(time_list), np.std(time_list)))

dataset_name = 'PROTEINS'
strategy = 'padding_pruning'

if 0:
    epoch_number = 500
    residual_type = 'raw'
    fold_list = range(1, 11)
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/AuGBert/' + strategy + '/' + dataset_name + '/'

    fold_result_dict = {}
    for fold in fold_list:
        result_obj.result_destination_file_name = dataset_name + '_' + str(fold) + '_' + str(epoch_number) + '_' + residual_type
        fold_result_dict[fold] = result_obj.load()

    x = range(epoch_number)

    plt.figure(figsize=(4, 3))
    for fold in fold_list:
        train_acc = [fold_result_dict[fold][i]['acc_train'] for i in x]
        plt.plot(x, train_acc, label=str(fold) + '-fold)')

    plt.xlim(0, epoch_number)
    plt.ylabel("training accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right", fontsize='small', ncol=2,)
    plt.show()

    plt.figure(figsize=(4, 3))
    for fold in fold_list:
        train_acc = [fold_result_dict[fold][i]['acc_test'] for i in x]
        plt.plot(x, train_acc, label=str(fold) + '-fold)')

    plt.xlim(0, epoch_number)
    plt.ylabel("testing accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right", fontsize='small', ncol=2,)
    plt.show()





