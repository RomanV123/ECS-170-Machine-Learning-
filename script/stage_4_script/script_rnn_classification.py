import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt
import re
from collections import Counter

np.random.seed(2)
torch.manual_seed(2)

# ---- Paths ----
DATA_PATH   = 'C:\\Users\\mothe\\Downloads\\ECS189G_Winter_2022_Source_Code_Template\\ECS189G_Winter_2022_Source_Code_Template\\data\\stage_4_data\\text_classification\\'
RESULT_PATH = 'C:\\Users\\mothe\\Downloads\\ECS189G_Winter_2022_Source_Code_Template\\ECS189G_Winter_2022_Source_Code_Template\\result\\stage_4_result\\'

VOCAB_SIZE   = 10000   # keep only the top-N most frequent words
MAX_SEQ_LEN  = 100     # pad/truncate every review to this many tokens
MAX_TRAIN    = 10000   # use a subset of training data to keep CPU training fast

# ---- Helper: load all .txt files from a folder ----
def load_folder(folder_path, label):
    texts, labels = [], []
    for fname in os.listdir(folder_path):
        if fname.endswith('.txt'):
            with open(os.path.join(folder_path, fname), 'r', encoding='utf-8') as f:
                texts.append(f.read())
            labels.append(label)
    return texts, labels

# ---- Helper: basic tokenizer ----
def tokenize(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)   # strip HTML tags like <br />
    text = re.sub(r'[^a-z\s]', ' ', text)  # keep only letters
    return text.split()

# ---- Helper: build vocabulary from training texts ----
def build_vocab(texts, vocab_size):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    # reserve index 0 for <PAD> and 1 for <UNK>
    most_common = counter.most_common(vocab_size - 2)
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in most_common:
        word_to_idx[word] = len(word_to_idx)
    return word_to_idx

# ---- Helper: encode a list of texts to fixed-length integer sequences ----
def encode(texts, word_to_idx, max_len):
    encoded = []
    for text in texts:
        tokens = tokenize(text)
        indices = [word_to_idx.get(t, 1) for t in tokens]  # 1 = <UNK>
        # Truncate to max_len
        indices = indices[:max_len]
        # Pad with 0 up to max_len
        indices += [0] * (max_len - len(indices))
        encoded.append(indices)
    return encoded

# ---- Load raw text files ----
print('Loading data...')
train_pos_texts, train_pos_labels = load_folder(DATA_PATH + 'train/pos', label=1)
train_neg_texts, train_neg_labels = load_folder(DATA_PATH + 'train/neg', label=0)
test_pos_texts,  test_pos_labels  = load_folder(DATA_PATH + 'test/pos',  label=1)
test_neg_texts,  test_neg_labels  = load_folder(DATA_PATH + 'test/neg',  label=0)

train_texts  = train_pos_texts  + train_neg_texts
train_labels = train_pos_labels + train_neg_labels
test_texts   = test_pos_texts   + test_neg_texts
test_labels  = test_pos_labels  + test_neg_labels

# Subsample training set for faster CPU training (balanced: half pos, half neg)
half = MAX_TRAIN // 2
train_texts  = train_pos_texts[:half]  + train_neg_texts[:half]
train_labels = train_pos_labels[:half] + train_neg_labels[:half]

print(f'  Train: {len(train_texts)} reviews | Test: {len(test_texts)} reviews')

# ---- Build vocabulary on training set only ----
print('Building vocabulary...')
word_to_idx = build_vocab(train_texts, VOCAB_SIZE)
print(f'  Vocab size: {len(word_to_idx)}')

# ---- Encode ----
train_X = encode(train_texts, word_to_idx, MAX_SEQ_LEN)
test_X  = encode(test_texts,  word_to_idx, MAX_SEQ_LEN)

loaded = {
    'train': {'X': train_X, 'y': train_labels},
    'test':  {'X': test_X,  'y': test_labels},
}

# ---- Build model ----
method_obj = Method_RNN_Classification(
    'RNN_Classification', '',
    vocab_size=len(word_to_idx),
    num_classes=2
)
method_obj.data = loaded

# ---- Result saver ----
result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = RESULT_PATH
result_obj.result_destination_file_name   = 'RNN_classification_result'
result_obj.fold_count = None

# ---- Run ----
print('************ Start: RNN Text Classification ************')
try:
    learned_result = method_obj.run()
except Exception as e:
    import traceback
    traceback.print_exc()
    raise

# ---- Evaluate ----
evaluate_obj = Evaluate_Accuracy('accuracy', '')
evaluate_obj.data = learned_result
accuracy, f1, precision, recall = evaluate_obj.evaluate()
print('************ Overall Performance ************')
print('Accuracy:', accuracy)
print('F1:', f1)
print('Precision:', precision)
print('Recall:', recall)
print('************ Finish ************')

# ---- Save result ----
result_obj.data = learned_result
result_obj.save()

# ---- Learning curves ----
epochs = list(range(len(method_obj.loss_list)))

plt.plot(epochs, method_obj.loss_list)
plt.title('RNN Classification Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(RESULT_PATH + 'RNN_classification_training_loss.png')
plt.clf()

plt.plot(epochs, method_obj.accuracy_list)
plt.title('RNN Classification Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(RESULT_PATH + 'RNN_classification_training_accuracy.png')
plt.clf()

print('Learning curves saved.')
