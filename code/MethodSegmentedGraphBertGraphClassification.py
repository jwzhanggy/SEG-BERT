import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers.modeling_bert import BertPreTrainedModel
from code.MethodGraphBert import MethodGraphBert

import time
import numpy as np

from code.EvaluateAcc import EvaluateAcc


BertLayerNorm = torch.nn.LayerNorm

class MethodSegmentedGraphBertGraphClassification(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    spy_tag = True
    fold = None
    strategy = None

    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config):
        super(MethodSegmentedGraphBertGraphClassification, self).__init__(config)
        self.config = config
        self.bert = MethodGraphBert(config)
        self.res_h = torch.nn.Linear(config.x_size**2, config.hidden_size)
        self.res_y = torch.nn.Linear(config.x_size**2, config.y_size)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.y_size)
        self.init_weights()

    def forward(self, x, d, w, wl, seg_count=None):
        residual_h, residual_y = self.residual_term(w)
        outputs = self.bert(x, d, w, wl, residual_h=residual_h)

        sequence_output = 0
        for i in range(self.config.k):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        segment_fusion_output = torch.zeros(size=[seg_count.size()[0], sequence_output.size()[1]])
        current_global_seg_index = 0
        for graph_index in range(seg_count.size()[0]):
            graph_seg_number = seg_count[graph_index].item()
            for seg_i in range(current_global_seg_index, current_global_seg_index + graph_seg_number):
                segment_fusion_output[graph_index] += sequence_output[seg_i]
            segment_fusion_output[graph_index] /= graph_seg_number
            current_global_seg_index += graph_seg_number

        labels = self.cls_y(segment_fusion_output)
        if residual_y is not None:
            labels += residual_y

        return F.log_softmax(labels, dim=1)

    def residual_term(self, w):
        batch, n, n = w.size()
        if self.config.residual_type == 'none':
            return None, None
        elif self.config.residual_type == 'raw':
            return self.res_h(w.view(batch, n*n)), self.res_y(w.view(batch, n*n))

    def train_model(self, max_epoch, fold):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        accuracy = EvaluateAcc('', '')

        for epoch in range(max_epoch):
            t_epoch_begin = time.time()

            # -------------------------
            self.train()
            optimizer.zero_grad()

            x, d, w, wl, y_true, segment_count_list = self.get_batch(self.data['train_idx'][fold])
            y_pred = self.forward(x, d, w, wl, segment_count_list)
            loss_train = F.cross_entropy(y_pred, y_true)
            accuracy.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            acc_train = accuracy.evaluate()

            loss_train.backward()
            optimizer.step()

            if self.spy_tag:
                self.eval()

                x, d, w, wl, y_true, segment_count_list = self.get_batch(self.data['test_idx'][fold])
                y_pred = self.forward(x, d, w, wl, segment_count_list)
                loss_test = F.cross_entropy(y_pred, y_true)
                accuracy.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                acc_test = accuracy.evaluate()

                self.learning_record_dict[epoch] = {'y_true': y_true, 'y_pred': y_pred,
                                                    'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
                                                    'loss_test': loss_test.item(), 'acc_test': acc_test.item(),
                                                    'time': time.time() - t_epoch_begin}

                # -------------------------
                if epoch % 10 == 0:
                    print('Fold: {:04d}'.format(fold),
                          'Epoch: {:04d}'.format(epoch + 1),
                          'loss_train: {:.4f}'.format(loss_train.item()),
                          'acc_train: {:.4f}'.format(acc_train.item()),
                          'loss_test: {:.4f}'.format(loss_test.item()),
                          'acc_test: {:.4f}'.format(acc_test.item()),
                          'time: {:.4f}s'.format(time.time() - t_epoch_begin))
            else:
                # -------------------------
                if epoch % 10 == 0:
                    print('Fold: {:04d}'.format(fold),
                          'Epoch: {:04d}'.format(epoch + 1),
                          'loss_train: {:.4f}'.format(loss_train.item()),
                          'acc_train: {:.4f}'.format(acc_train.item()),
                          'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin) + ', best testing performance {: 4f}'.format(np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])) + ', minimun loss {: 4f}'.format(np.min([self.learning_record_dict[epoch]['loss_test'] for epoch in self.learning_record_dict])))
        return time.time() - t_begin, np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])

    def get_batch(self, id_list):
        x = []
        d = []
        w = []
        wl = []
        y = []
        segment_count_list = []
        for id in id_list:
            if self.strategy == 'isolated_segment':
                seg_count = 0
                for segment_start in range(0, self.config.x_size, self.config.k):
                    idx_list = range(segment_start, segment_start+self.config.k)
                    x_temp = [self.data['processed_graph_data'][id]['tag'][idx] for idx in idx_list]
                    d_temp = [self.data['processed_graph_data'][id]['degree'][idx] for idx in idx_list]
                    w_temp = [self.data['processed_graph_data'][id]['weight'][idx] for idx in idx_list]
                    wl_temp = [self.data['processed_graph_data'][id]['wl_tag'][idx] for idx in idx_list]
                    if np.sum(d_temp) == 0 and np.sum(w_temp) == 0 and np.sum(wl_temp) == 0: continue
                    x.append(x_temp)
                    d.append(d_temp)
                    w.append(w_temp)
                    wl.append(wl_temp)
                    seg_count += 1
                segment_count_list.append(seg_count)
                y.append(self.data['processed_graph_data'][id]['y'])
            # elif self.strategy == 'augmented_segment':
            #     k = int(self.config.k/3)
            #     for segment_start in range(0, self.config.x_size, k):
            #         idx_list = range(segment_start, segment_start+k)
            #         res = [i for i in range(self.config.x_size) if i not in idx_list]
            #         x.append(self.data['processed_graph_data'][id]['tag'])
            #         d.append(self.data['processed_graph_data'][id]['degree'])
            #         w.append(self.data['processed_graph_data'][id]['weight'])
            #         wl.append(self.data['processed_graph_data'][id]['wl_tag'])
            #         y.append(self.data['processed_graph_data'][id]['y'])
            #         context_idx_list.append(res)
        return torch.LongTensor(x), torch.LongTensor(d), torch.FloatTensor(w), torch.LongTensor(wl), torch.LongTensor(y), torch.LongTensor(segment_count_list)

    def run(self):
        self.train_model(self.max_epoch, self.fold)
        return self.learning_record_dict
