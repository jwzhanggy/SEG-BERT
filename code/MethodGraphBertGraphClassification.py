import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers.modeling_bert import BertPreTrainedModel
from code.MethodGraphBert import MethodGraphBert

import time
import numpy as np

from code.EvaluateAcc import EvaluateAcc


BertLayerNorm = torch.nn.LayerNorm

class MethodGraphBertGraphClassification(BertPreTrainedModel):
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
        super(MethodGraphBertGraphClassification, self).__init__(config)
        self.config = config
        self.bert = MethodGraphBert(config)
        self.res_h = torch.nn.Linear(config.x_size**2, config.hidden_size)
        self.res_y = torch.nn.Linear(config.x_size**2, config.y_size)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.y_size)
        self.init_weights()

    def forward(self, x, d, w, wl, context_idx=None):
        residual_h, residual_y = self.residual_term(w)
        outputs = self.bert(x, d, w, wl, residual_h=residual_h, context_stop_grad_idx=context_idx)

        sequence_output = 0
        for i in range(self.config.k):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        labels = self.cls_y(sequence_output)
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

            x, d, w, wl, y_true, context_idx_list = self.get_batch(self.data['train_idx'][fold])
            y_pred = self.forward(x, d, w, wl, context_idx_list)
            loss_train = F.cross_entropy(y_pred, y_true)
            accuracy.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            acc_train = accuracy.evaluate()

            loss_train.backward()
            optimizer.step()

            if self.spy_tag:
                self.eval()

                x, d, w, wl, y_true, context_idx_list = self.get_batch(self.data['test_idx'][fold])
                y_pred = self.forward(x, d, w, wl, context_idx_list)
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
        context_idx_list = []
        for id in id_list:
            x.append(self.data['processed_graph_data'][id]['tag'])
            d.append(self.data['processed_graph_data'][id]['degree'])
            w.append(self.data['processed_graph_data'][id]['weight'])
            wl.append(self.data['processed_graph_data'][id]['wl_tag'])
            y.append(self.data['processed_graph_data'][id]['y'])
        return torch.LongTensor(x), torch.LongTensor(d), torch.FloatTensor(w), torch.LongTensor(wl), torch.LongTensor(y), torch.LongTensor(context_idx_list)

    def run(self):
        self.train_model(self.max_epoch, self.fold)
        return self.learning_record_dict
