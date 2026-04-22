from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np

class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.fc_layer_1 = nn.Linear(784, 512)
        self.activation_func_1 = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc_layer_2 = nn.Linear(512, 256)
        self.activation_func_2 = nn.LeakyReLU()
        self.fc_layer_3 = nn.Linear(256, 128)
        self.activation_func_3 = nn.LeakyReLU()
        self.fc_layer_4 = nn.Linear(128, 10)
        self.loss_list = []
        self.accuracy_list = []

    def forward(self, x):
        h = self.activation_func_1(self.fc_layer_1(x))
        h = self.dropout(h)
        h1 = self.activation_func_2(self.fc_layer_2(h))
        h1 = self.dropout(h1)
        h2 = self.activation_func_3(self.fc_layer_3(h1))
        h2 = self.dropout(h2)
        y_pred = self.fc_layer_4(h2)
        return y_pred

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, momentum=0.9) 
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # Convert data to tensors ONCE before the loop (big speedup)
        X_tensor = torch.FloatTensor(np.array(X))
        y_true = torch.LongTensor(np.array(y))

        for epoch in range(self.max_epoch):
            y_pred = self.forward(X_tensor)
            train_loss = loss_function(y_pred, y_true)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                acc, _, _, _ = accuracy_evaluator.evaluate()
                self.loss_list.append(train_loss.item())
                self.accuracy_list.append(acc)
                print('Epoch:', epoch, 'Accuracy:', acc, 'Loss:', train_loss.item())

    def test(self, X):
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}