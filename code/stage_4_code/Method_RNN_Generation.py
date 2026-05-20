from code.base_class.method import method
import torch
from torch import nn
import numpy as np


class Method_RNN_Generation(method, nn.Module):
    data = None           # expects {'text': list_of_strings, 'word_to_idx': dict, 'idx_to_word': dict}
    max_epoch = 20
    learning_rate = 1e-3
    sequence_length = 20  # how many words to look at per training step

    def __init__(self, mName, mDescription, vocab_size=10000):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.vocab_size = vocab_size

        # ---- Embedding layer ----
        # Maps each word index → a dense vector of size 128
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        
        self.rnn = nn.LSTM(
            input_size = 128,  # must match embedding dimension
            hidden_size = 256, # you can choose this
            num_layers = 2,    # you can choose this
            batch_first = True,# input shape should be (batch, seq_len, features)
            dropout = 0.3      # you can choose this
        )

        self.fc = nn.Linear(256, vocab_size)  # maps from hidden_size to vocab_size
        self.dropout = nn.Dropout(0.3)
        self.dropout = nn.Dropout(0.3)

        self.loss_list = []

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        x: LongTensor of shape (batch_size, seq_len)
        Returns logits of shape (batch_size, seq_len, vocab_size)
        """
        # Step 1: embed the input indices
        #   embedded shape: (batch_size, seq_len, 128)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        out, (h_n, c_n) = self.rnn(embedded)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


    # ------------------------------------------------------------------
    def fit(self, token_ids):
        """
        token_ids: flat list/array of integer word indices for the entire corpus.
        Trains the model to predict the next word at each position.
        """
        nn.Module.train(self)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        token_ids = np.array(token_ids, dtype=np.int64)
        n = len(token_ids)
        batch_size = 64
        seq_len = self.sequence_length

        for epoch in range(self.max_epoch):
            epoch_loss = 0.0
            num_batches = 0

            # Slide through the corpus in non-overlapping windows
            for start in range(0, n - seq_len - 1, seq_len):
                # Build a batch of (input, target) pairs
                # X: words 0..seq_len-1, y: words 1..seq_len (next-word targets)
                X_batch, y_batch = [], []
                for b in range(batch_size):
                    idx = start + b * seq_len
                    if idx + seq_len + 1 > n:
                        break
                    X_batch.append(token_ids[idx: idx + seq_len])
                    y_batch.append(token_ids[idx + 1: idx + seq_len + 1])

                if not X_batch:
                    continue

                X_tensor = torch.LongTensor(np.array(X_batch))
                y_tensor = torch.LongTensor(np.array(y_batch))

                logits = self.forward(X_tensor)   # (batch, seq_len, vocab_size)

                # Reshape for CrossEntropyLoss: (batch*seq_len, vocab_size) and (batch*seq_len,)
                loss = loss_function(
                    logits.reshape(-1, self.vocab_size),
                    y_tensor.reshape(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # prevent exploding gradients
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            self.loss_list.append(avg_loss)
            print(f'Epoch: {epoch}  Loss: {avg_loss:.4f}')

    # ------------------------------------------------------------------
    def generate(self, seed_tokens, word_to_idx, idx_to_word, num_words=50, temperature=1.0):
        """
        Generate text starting from a seed sequence of word indices.

        seed_tokens : list of int  — starting word indices
        word_to_idx : dict         — word → index
        idx_to_word : dict         — index → word
        num_words   : int          — how many new words to generate
        temperature : float        — > 1 makes output more random, < 1 makes it more conservative
        """
        nn.Module.eval(self)
        generated = list(seed_tokens)

        with torch.no_grad():
            for _ in range(num_words):
                # Use the last `sequence_length` tokens as context
                context = generated[-self.sequence_length:]
                x = torch.LongTensor([context])          # shape (1, seq_len)
                logits = self.forward(x)                  # (1, seq_len, vocab_size)
                next_logits = logits[0, -1, :]            # last time step → (vocab_size,)

                # Apply temperature scaling, then sample from the distribution
                probs = torch.softmax(next_logits / temperature, dim=0).numpy()
                next_idx = np.random.choice(len(probs), p=probs)
                generated.append(next_idx)

        words = [idx_to_word.get(i, '<UNK>') for i in generated]
        return ' '.join(words)

    # ------------------------------------------------------------------
    def run(self):
        """
        Trains on self.data['token_ids'] and generates a sample.
        self.data must contain:
            token_ids   : list of int (entire corpus encoded)
            word_to_idx : dict
            idx_to_word : dict
        """
        print('method running...')
        print('--start training...')
        self.fit(self.data['token_ids'])

        print('--generating sample text...')
        # Use the first sequence_length tokens as the seed
        seed = self.data['token_ids'][:self.sequence_length]
        sample = self.generate(
            seed_tokens=seed,
            word_to_idx=self.data['word_to_idx'],
            idx_to_word=self.data['idx_to_word'],
            num_words=50,
            temperature=0.8
        )
        print('Generated text:\n', sample)
        return {'generated': sample, 'loss_list': self.loss_list}
