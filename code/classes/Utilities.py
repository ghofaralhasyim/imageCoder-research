from __future__ import absolute_import

from classes.Dataset import *

class Utilities:

    @staticmethod
    def preprocess_data(input_path, validation_split):

        dataset = Dataset(input_path)
        training_path, validation_path = dataset.split_datasets(validation_split)
        dataset.build_data(training_path, validation_path)

        return training_path, validation_path
    
    