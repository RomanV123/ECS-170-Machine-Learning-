from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_RNN_Classification(method, nn.Module):
    data = None
    max_epoch = 20
    learning_rate = 1e-3

    # ------------------------------------------------------------------ #
    # You need to set these before initializing:                           #
    #   vocab_size  - total number of unique words in your vocabulary      #
    #   num_classes - 2 for binary classification (pos / neg)             #
    # ------------------------------------------------------------------ #

    def __init__(self, mName, mDescription, vocab_size=10000, num_classes=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

   
        self.embedding_dim = 128
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)

 
        self.rnn = nn.GRU(
            input_size = self.embedding_dim,  # must match embedding dimension
            hidden_size = 256,                # you can choose this
            num_layers = 3,                   # you can choose this
            batch_first = True,               # input shape should be (batch, seq_len, features
            dropout = 0.3                    # you can choose this
        )
        self.fc = nn.Linear(256, num_classes)  # maps from hidden_size to num_classes
        self.dropout = nn.Dropout(0.5)
        self.loss_list = []
        self.accuracy_list = []

    def forward(self, x):

        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        out, h_n = self.rnn(embedded)
        last_hidden = h_n[-1]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits
        

    def fit(self, X, y):
        nn.Module.train(self)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_array = np.array(X)
        y_array = np.array(y)
        batch_size = 32
        n = len(X_array)

        for epoch in range(self.max_epoch):
            indices = np.random.permutation(n)
            epoch_loss = 0.0

            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = torch.LongTensor(X_array[batch_idx])
                y_batch = torch.LongTensor(y_array[batch_idx])

                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch % 2 == 0:
                sample_idx = indices[:200]
                X_sample = torch.LongTensor(X_array[sample_idx])
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
        batch_size = 32
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_array), batch_size):
                X_batch = torch.LongTensor(X_array[start:start + batch_size])
                preds.append(self.forward(X_batch).max(1)[1])
        return torch.cat(preds)

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
