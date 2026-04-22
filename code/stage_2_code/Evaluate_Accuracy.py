from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        accuracy  = accuracy_score(true_y, pred_y)
        f1        = f1_score(true_y, pred_y, average='weighted')
        precision = precision_score(true_y, pred_y, average='weighted')
        recall    = recall_score(true_y, pred_y, average='weighted')

        return accuracy, f1, precision, recall