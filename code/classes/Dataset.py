import os
import shutil
import hashlib
import numpy as np

from classes.Preprocess import *
from classes.Utilities import *

from keras.preprocessing.text import Tokenizer, one_hot
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

VOCABULARY_PATH         = '../vocabulary.vocab'
TRAINING_SET_NAME       = "training_set"
VALIDATION_SET_NAME     = "validation_set"
BATCH_SIZE              = 64

class Dataset:
    def __init__(self, input_path):
        self.input_path = input_path
        self.test_set_folder = None

    def split_datasets(self, validation_split):
        sample_ids = self.populate_data()
        print("Total samples: ", len(sample_ids))

        np.random.shuffle(sample_ids)
        val_count = int(validation_split * len(sample_ids))
        train_count = len(sample_ids) - val_count
        print("Splitting datasets, training samples: {}, validation samples: {}".format(train_count, val_count))
        train_set_ids, val_set_ids = self.split_paths(sample_ids, train_count, val_count)

        training_path = "{}/{}".format(os.path.dirname(self.input_path), TRAINING_SET_NAME)
        validation_path = "{}/{}".format(os.path.dirname(self.input_path), VALIDATION_SET_NAME)

        self.delete_folder(training_path)
        self.delete_folder(validation_path)

        if not os.path.exists(training_path): os.makedirs(training_path)
        if not os.path.exists(validation_path): os.makedirs(validation_path)

        self.allocate_to_folder(train_set_ids, training_path)
        self.allocate_to_folder(val_set_ids, validation_path)
        
        return training_path, validation_path
    
    def build_data(self, training_path, validation_path):
        train_img_preprocessor = DataPrepocessor()
        train_img_preprocessor.build_image_dataset(training_path)
        val_img_preprocessor = DataPrepocessor()
        val_img_preprocessor.build_image_dataset(validation_path)
    
    def populate_data(self):
        all_sample_ids = []
        full_path = os.path.realpath(self.input_path)
        for f in os.listdir(full_path):
            if f.find(".gui") != -1:
                file_name = f[:f.find(".gui")]
                if os.path.isfile("{}/{}.png".format(self.input_path, file_name)):
                    all_sample_ids.append(file_name)
        return all_sample_ids
    
    def split_paths(self, sample_ids, train_count, val_count):
        train_set = []
        val_set = []
        hashes = []
        for sample_id in sample_ids:
            f = open("{}/{}.gui".format(self.input_path, sample_id), 'r', encoding='utf-8')

            with f:
                chars = ""
                for line in f:
                    chars += line
                content_hash = chars.replace(" ", "").replace("\n", "")
                content_hash = hashlib.sha256(content_hash.encode('utf-8')).hexdigest()

                if len(val_set) == val_count:
                    train_set.append(sample_id)
                else:
                    is_unique = True
                    for h in hashes:
                        if h is content_hash:
                            is_unique = False
                            break

                    if is_unique:
                        val_set.append(sample_id)
                    else:
                        train_set.append(sample_id)

                hashes.append(content_hash)

        assert len(val_set) == val_count

        return train_set, val_set
    
    def allocate_to_folder(self, sample_ids, output_folder):
        copied_count = 0
        for sample_id in sample_ids:
            sample_id_png_path = "{}/{}.png".format(self.input_path, sample_id)
            sample_id_gui_path = "{}/{}.gui".format(self.input_path, sample_id)
            if os.path.exists(sample_id_png_path) and os.path.exists(sample_id_gui_path):
                output_png_path = "{}/{}.png".format(output_folder, sample_id)
                output_gui_path = "{}/{}.gui".format(output_folder, sample_id)
                shutil.copyfile(sample_id_png_path, output_png_path)
                shutil.copyfile(sample_id_gui_path, output_gui_path)
                copied_count += 1
        print("Moved {} files from {} to {}".format(copied_count, self.input_path, output_folder))

    def delete_folder(self, folder_to_delete):
        if os.path.exists(folder_to_delete):
            shutil.rmtree(folder_to_delete)
            print("Deleted existing folder: {}".format(folder_to_delete))
