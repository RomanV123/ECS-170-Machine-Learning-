import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Method_GCN import Method_GCN
import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(2)
torch.manual_seed(2)

# ---- Paths ----
DATA_PATH   = 'C:\\Users\\mothe\\Downloads\\ECS189G_Winter_2022_Source_Code_Template\\ECS189G_Winter_2022_Source_Code_Template\\data\\stage_5_data\\'
RESULT_PATH = 'C:\\Users\\mothe\\Downloads\\ECS189G_Winter_2022_Source_Code_Template\\ECS189G_Winter_2022_Source_Code_Template\\result\\stage_5_result\\'
os.makedirs(RESULT_PATH, exist_ok=True)

# ---- Dataset configs: (name, num_classes, num_layers, epochs, dropout, lr) ----
# Cora:     2 layers, 400 epochs, dropout 0.3, lr 5e-3
# Citeseer: 2 layers, 500 epochs, dropout 0.6, lr 5e-4 — slower lr + higher dropout to close train/test gap
# Pubmed:   3 layers, 500 epochs, dropout 0.3, lr 5e-3
DATASETS = [
    ('cora',     7, 2, 400, 0.3, 5e-3),
    ('citeseer', 6, 2, 500, 0.6, 5e-4),
    ('pubmed',   3, 3, 500, 0.3, 5e-3),
]

all_results = {}

for dataset_name, num_classes, num_layers, max_epoch, dropout, lr in DATASETS:
    print(f'\n{"="*50}')
    print(f'Dataset: {dataset_name.upper()}')
    print(f'{"="*50}')

    # ---- Load data ----
    loader = Dataset_Loader(dName=dataset_name, dDescription='')
    loader.dataset_name = dataset_name
    loader.dataset_source_folder_path = DATA_PATH + dataset_name
    data = loader.load()

    input_dim = data['graph']['X'].shape[1]
    print(f'Nodes: {data["graph"]["X"].shape[0]} | Features: {input_dim} | Classes: {num_classes}')

    # ---- Build model ----
    hidden_dim = 128
    model = Method_GCN(
        mName=f'GCN_{dataset_name}',
        mDescription='',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout
    )
    model.max_epoch = max_epoch
    model.learning_rate = lr
    model.data = data

    # ---- Run ----
    print(f'************ Start: GCN on {dataset_name} ************')
    try:
        result = model.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

    print(f'************ Overall Performance — {dataset_name} ************')
    print(f'Accuracy:  {result["acc"]:.4f}')
    print(f'F1:        {result["f1"]:.4f}')
    print(f'Precision: {result["precision"]:.4f}')
    print(f'Recall:    {result["recall"]:.4f}')
    print(f'************ Finish ************')

    all_results[dataset_name] = result

    # ---- Learning curves ----
    epochs = [i * 10 for i in range(len(result['loss_list']))]

    plt.figure()
    plt.plot(epochs, result['loss_list'], color='#2196F3', linewidth=2)
    plt.title(f'GCN {dataset_name.capitalize()} — Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULT_PATH + f'{dataset_name}_training_loss.png', dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, result['accuracy_list'], color='#4CAF50', linewidth=2)
    plt.title(f'GCN {dataset_name.capitalize()} — Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULT_PATH + f'{dataset_name}_training_accuracy.png', dpi=150)
    plt.close()

    print(f'Learning curves saved for {dataset_name}.')

# ---- Summary table ----
print('\n\n============ SUMMARY ============')
print(f'{"Dataset":<12} {"Accuracy":<12} {"F1":<12} {"Precision":<12} {"Recall":<12}')
print('-' * 60)
for name, res in all_results.items():
    print(f'{name:<12} {res["acc"]:<12.4f} {res["f1"]:<12.4f} {res["precision"]:<12.4f} {res["recall"]:<12.4f}')
