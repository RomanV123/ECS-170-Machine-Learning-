import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
import re
from collections import Counter

np.random.seed(2)
torch.manual_seed(2)

# ---- Paths ----
DATA_PATH   = 'C:\\Users\\mothe\\Downloads\\ECS189G_Winter_2022_Source_Code_Template\\ECS189G_Winter_2022_Source_Code_Template\\data\\stage_4_data\\text_generation\\data'
RESULT_PATH = 'C:\\Users\\mothe\\Downloads\\ECS189G_Winter_2022_Source_Code_Template\\ECS189G_Winter_2022_Source_Code_Template\\result\\stage_4_result\\'

VOCAB_SIZE = 5000   # keep top-N words

# ---- Helper: basic tokenizer ----
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text.split()

# ---- Load jokes CSV ----
print('Loading data...')
jokes = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        joke = row['Joke'].strip()
        if joke:
            jokes.append(joke)

print(f'  Loaded {len(jokes)} jokes')

# Concatenate all jokes into one big corpus
full_text = ' <END> '.join(jokes)   # <END> marks the boundary between jokes

# ---- Build vocabulary ----
print('Building vocabulary...')
tokens = tokenize(full_text)
counter = Counter(tokens)
# Reserve 0 for <PAD>, 1 for <UNK>
most_common = counter.most_common(VOCAB_SIZE - 2)
word_to_idx = {'<PAD>': 0, '<UNK>': 1}
for word, _ in most_common:
    word_to_idx[word] = len(word_to_idx)
idx_to_word = {v: k for k, v in word_to_idx.items()}
print(f'  Vocab size: {len(word_to_idx)}')

# ---- Encode entire corpus ----
token_ids = [word_to_idx.get(t, 1) for t in tokens]
print(f'  Total tokens: {len(token_ids)}')

# ---- Build model ----
method_obj = Method_RNN_Generation(
    'RNN_Generation', '',
    vocab_size=len(word_to_idx)
)
method_obj.data = {
    'token_ids':   token_ids,
    'word_to_idx': word_to_idx,
    'idx_to_word': idx_to_word,
}

# ---- Run ----
print('************ Start: RNN Text Generation ************')
try:
    result = method_obj.run()
except Exception as e:
    import traceback
    traceback.print_exc()
    raise

print('************ Generated Sample (temp=0.8) ************')
print(result['generated'])
print('************ Finish ************')

# ---- Save generated text ----
os.makedirs(RESULT_PATH, exist_ok=True)
with open(RESULT_PATH + 'RNN_generation_output.txt', 'w', encoding='utf-8') as f:
    f.write(result['generated'])
print('Generated text saved.')

# ---- Temperature ablation ----
seed = method_obj.data['token_ids'][:method_obj.sequence_length]

for temp in [0.5, 1.0, 1.5]:
    sample = method_obj.generate(
        seed_tokens=seed,
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        num_words=50,
        temperature=temp
    )
    print(f'\n--- Temperature {temp} ---')
    print(sample)
    with open(RESULT_PATH + f'RNN_generation_temp{temp}.txt', 'w', encoding='utf-8') as f:
        f.write(sample)

print('\nTemperature ablation saved.')

# ---- 3-word seed examples for report ----
def generate_from_words(words, num_words=30, temperature=0.8):
    seed = [word_to_idx.get(w, 1) for w in words]  # 1 = <UNK> if not in vocab
    return method_obj.generate(seed, word_to_idx, idx_to_word, num_words, temperature)

print('\n************ 3-Word Seed Examples ************')
seeds = [
    ['what', 'did', 'the'],
    ['why', 'did', 'the'],
    ['how', 'many', 'people'],
    ['pizza', 'loves', 'coffee'],   # out-of-dataset words
]
for words in seeds:
    output = generate_from_words(words)
    print(f'\nSeed: "{" ".join(words)}"')
    print(output)
    fname = '_'.join(words) + '.txt'
    with open(RESULT_PATH + 'seed_' + fname, 'w', encoding='utf-8') as f:
        f.write(output)

print('\n3-word seed examples saved.')

# ---- Learning curve ----
plt.plot(result['loss_list'])
plt.title('RNN Generation Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(RESULT_PATH + 'RNN_generation_training_loss.png')
plt.clf()
print('Learning curve saved.')
