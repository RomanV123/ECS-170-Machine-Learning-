import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(2)
torch.manual_seed(2)

DATA_PATH = 'C:\\Users\\mothe\\Downloads\\ECS189G_Winter_2022_Source_Code_Template\\ECS189G_Winter_2022_Source_Code_Template\\data\\stage_3_data\\'
RESULT_PATH = 'C:\\Users\\mothe\\Downloads\\ECS189G_Winter_2022_Source_Code_Template\\ECS189G_Winter_2022_Source_Code_Template\\result\\stage_3_result\\'

# ---- Choose dataset here: 'MNIST', 'ORL', or 'CIFAR' ----
DATASET = 'ORL'
  
# Settings per dataset
config = {
    'MNIST': {'in_channels': 1, 'num_classes': 10},
    'ORL':   {'in_channels': 1, 'num_classes': 40},
    'CIFAR': {'in_channels': 3, 'num_classes': 10},
}

# ---- Load data directly from pickle ----
print('loading data...')
f = open(DATA_PATH + DATASET, 'rb')
raw = pickle.load(f)
f.close()

def process(instances, dataset_name):
    X, y = [], []
    for instance in instances:
        img = np.array(instance['image'], dtype=np.float32) / 255.0
        if dataset_name == 'MNIST':
            img = img.reshape(1, 28, 28)
        elif dataset_name == 'ORL':
            img = img[:, :, 0].reshape(1, 112, 92)
        elif dataset_name == 'CIFAR':
            img = img.transpose(2, 0, 1)
        X.append(img)
        label = instance['label']
        if dataset_name == 'ORL':
            label = label - 1  # shift from 1-40 to 0-39
        y.append(label)
    return X, y

train_X, train_y = process(raw['train'], DATASET)
test_X,  test_y  = process(raw['test'],  DATASET)
loaded = {'train': {'X': train_X, 'y': train_y}, 'test': {'X': test_X, 'y': test_y}}

# ---- Build model ----
cfg = config[DATASET]
method_obj = Method_CNN('CNN', '', in_channels=cfg['in_channels'], num_classes=cfg['num_classes'])
method_obj.data = loaded

# ---- Result saver ----
result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = RESULT_PATH
result_obj.result_destination_file_name = DATASET + '_prediction_result'
result_obj.fold_count = None

# ---- Run ----
print('************ Start:', DATASET, '************')
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
epochs = list(range(0, method_obj.max_epoch, 5))

plt.plot(epochs, method_obj.loss_list)
plt.title(DATASET + ' CNN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(RESULT_PATH + DATASET + '_training_loss.png')
plt.clf()

plt.plot(epochs, method_obj.accuracy_list)
plt.title(DATASET + ' CNN Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(RESULT_PATH + DATASET + '_training_accuracy.png')
plt.clf()

print('Learning curves saved.')
