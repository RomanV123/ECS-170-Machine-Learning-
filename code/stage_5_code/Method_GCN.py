from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # H = A_hat * X * W
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output


class Method_GCN(method, nn.Module):
    data = None
    max_epoch = 300
    learning_rate = 5e-3

    def __init__(self, mName, mDescription, input_dim=1433, hidden_dim=128, num_classes=7, num_layers=2, dropout=0.3):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.num_layers = num_layers

        self.gc1 = GCNLayer(input_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, hidden_dim)        # only used if num_layers == 3
        self.gc_out = GCNLayer(hidden_dim, num_classes)    # always the final layer
        self.dropout = nn.Dropout(dropout)

        self.loss_list = []
        self.accuracy_list = []

    def forward(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        h = self.dropout(h)
        if self.num_layers == 3:
            h = F.relu(self.gc2(h, adj))
            h = self.dropout(h)
        h = self.gc_out(h, adj)
        return F.log_softmax(h, dim=1)

    def fit(self):
        nn.Module.train(self)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        loss_function = nn.NLLLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X   = self.data['graph']['X']
        y   = self.data['graph']['y']
        adj = self.data['graph']['utility']['A']
        idx_train = self.data['train_test_val']['idx_train']

        for epoch in range(self.max_epoch):
            optimizer.zero_grad()
            logits = self.forward(X, adj)
            loss = loss_function(logits[idx_train], y[idx_train])
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                with torch.no_grad():
                    pred_y = logits[idx_train].max(1)[1]
                    accuracy_evaluator.data = {'true_y': y[idx_train], 'pred_y': pred_y}
                    acc, _, _, _ = accuracy_evaluator.evaluate()
                    self.loss_list.append(loss.item())
                    self.accuracy_list.append(acc)
                    print(f'Epoch: {epoch}  Loss: {loss.item():.4f}  Train Accuracy: {acc:.4f}')

    def test(self):
        nn.Module.eval(self)
        accuracy_evaluator = Evaluate_Accuracy('test evaluator', '')

        X   = self.data['graph']['X']
        y   = self.data['graph']['y']
        adj = self.data['graph']['utility']['A']
        idx_test = self.data['train_test_val']['idx_test']

        with torch.no_grad():
            logits = self.forward(X, adj)
            pred_y = logits[idx_test].max(1)[1]

        accuracy_evaluator.data = {'true_y': y[idx_test], 'pred_y': pred_y}
        acc, f1, precision, recall = accuracy_evaluator.evaluate()
        return acc, f1, precision, recall

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit()
        print('--start testing...')
        acc, f1, precision, recall = self.test()
        return {'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall,
                'loss_list': self.loss_list, 'accuracy_list': self.accuracy_list}
