from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np

class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 100
    learning_rate = 1e-4

    def __init__(self, mName, mDescription, in_channels=3, num_classes=10):
        #Convolution + Relu, Convolution + Relu, pooling, then flatten
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv_layer_1 = nn.Conv2d(in_channels, out_channels = 32, kernel_size = 3)
        self.activation_func_1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.conv_layer_2 = nn.Conv2d(in_channels = 32 , out_channels = 32, kernel_size = 3)
        self.activation_func_2 = nn.LeakyReLU() 
        self.bn2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_layer_3 = nn.Conv2d(in_channels = 32 , out_channels = 64, kernel_size = 3)
        self.activation_func_3 = nn.LeakyReLU() 
        self.bn3 = nn.BatchNorm2d(64)   
        self.conv_layer_4 = nn.Conv2d(in_channels = 64 , out_channels = 64, kernel_size = 3)
        self.activation_func_4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(64)
        self.max_pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.LazyLinear(512)
        self.activation_func_5 = nn.LeakyReLU()
        self.fc2 = nn.Linear(512,128)
        self.activation_func_6 = nn.LeakyReLU()
        self.fc_layer_3 = nn.Linear(128, num_classes)
        self.loss_list = []
        self.accuracy_list = []


    def forward(self, x):
        h = self.conv_layer_1(x)
        h = self.bn1(h)
        h = self.activation_func_1(h)
        h = self.conv_layer_2(h)
        h = self.bn2(h)
        h = self.activation_func_2(h)
        h = self.max_pool1(h)
        h = self.dropout(h)
        
        h = self.conv_layer_3(h)
        h = self.bn3(h)
        h = self.activation_func_3(h)
        h = self.conv_layer_4(h)
        h = self.bn4(h)
        h = self.activation_func_4(h)
        h = self.max_pool4(h)
        h = self.dropout(h)

        h = h.reshape(h.size(0), -1)  # flatten         

        h = self.fc1(h)
        h = self.activation_func_5(h)
        h = self.dropout(h)
        h = self.fc2(h)
    
        h = self.activation_func_6(h)
        y_pred = self.fc_layer_3(h)
        return y_pred

    

    def fit(self, X, y):
        nn.Module.train(self)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_array = np.array(X)
        y_array = np.array(y)
        batch_size = 64
        n = len(X_array)

        for epoch in range(self.max_epoch):
            # shuffle each epoch
            indices = np.random.permutation(n)
            epoch_loss = 0.0

            # mini-batch loop
            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = torch.FloatTensor(X_array[batch_idx])
                y_batch = torch.LongTensor(y_array[batch_idx])

                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch % 5 == 0:
                # evaluate on a small sample to track accuracy
                sample_idx = indices[:500]
                X_sample = torch.FloatTensor(X_array[sample_idx])
                y_sample = torch.LongTensor(y_array[sample_idx])
                with torch.no_grad():
                    y_sample_pred = self.forward(X_sample)
                accuracy_evaluator.data = {'true_y': y_sample, 'pred_y': y_sample_pred.max(1)[1]}
                acc, _, _, _ = accuracy_evaluator.evaluate()
                avg_loss = epoch_loss / (n // batch_size)
                self.loss_list.append(avg_loss)
                self.accuracy_list.append(acc)
                print('Epoch:', epoch, 'Accuracy:', acc, 'Loss:', avg_loss)

    def test(self, X):
        nn.Module.eval(self)
        X_array = np.array(X)
        batch_size = 64
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_array), batch_size):
                X_batch = torch.FloatTensor(X_array[start:start + batch_size])
                preds.append(self.forward(X_batch).max(1)[1])
        return torch.cat(preds)

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}