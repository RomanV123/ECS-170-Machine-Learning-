import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- object initialization section ---------------
    data_obj = Dataset_Loader('MNIST', '')
    data_obj.dataset_source_folder_path = 'C:\\Users\\mothe\\Downloads\\ECS189G_Winter_2022_Source_Code_Template\\ECS189G_Winter_2022_Source_Code_Template\\data\\stage_2_data\\'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'C:\\Users\\mothe\\Downloads\\ECS189G_Winter_2022_Source_Code_Template\\ECS189G_Winter_2022_Source_Code_Template\\result\\stage_2_result\\'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    (accuracy, f1, precision, recall), _ = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy:', (accuracy))
    print('MLP F1:', f1)
    print('MLP Precision:', precision)
    print('MLP Recall:', recall)
    print('************ Finish ************')

    epoch = [0,100,200,300,400]
    
    loss_list = method_obj.loss_list
    accuracy_list = method_obj.accuracy_list

    plt.plot(epoch,loss_list)
    plt.title('MLP Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('result/stage_2_result/MLP_training_loss_SGD.png')
    plt.clf()


    plt.plot(epoch,accuracy_list)
    plt.title('MLP Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('result/stage_2_result/MLP_training_accuracy_SGD.png')
    plt.clf()

