'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting

class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):

        # load train data
        self.dataset.dataset_source_file_name = 'train.csv'
        train_data = self.dataset.load()

        # load test data
        self.dataset.dataset_source_file_name = 'test.csv'
        test_data = self.dataset.load()

        # run method
        self.method.data = {
            'train': {'X': train_data['X'], 'y': train_data['y']},
            'test':  {'X': test_data['X'],  'y': test_data['y']}
        }
        learned_result = self.method.run()

        # save result
        self.result.data = learned_result
        self.result.save()

        # evaluate
        self.evaluate.data = learned_result
        return self.evaluate.evaluate(), None