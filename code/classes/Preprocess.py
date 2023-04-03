import os
import sys
import shutil

import numpy as np
from PIL import Image
import cv2
from keras.preprocessing.image import ImageDataGenerator

class DataPrepocessor:

    def __init__(self):
        pass

    def build_image_dataset(self, input_path):

        print("Converting images {}".format(input_path))
        resized_img_arrays, sample_ids = self.get_resized_images(input_path)
        self.save_resized_img_arrays(resized_img_arrays, sample_ids, input_path)

    def get_img_features(self, png_path):
        img_features = self.resize_img1(png_path)
        assert(img_features.shape == (256,256,3))
        return img_features
    
    def get_resized_images(self, pngs_input_folder):
        all_files = os.listdir(pngs_input_folder)
        image_files = [f for f in all_files if f.find(".png") != -1]
        images = []
        labels = []
        for img_file_path in image_files:
            img_path = "{}/{}".format(pngs_input_folder, img_file_path)
            sample_id = img_file_path[:img_file_path.find('.png')]
            resized_img_arr = self.resize_img(img_path)
            images.append(resized_img_arr)
            labels.append(sample_id)
        return np.array(images), np.array(labels)
    
    def resize_img_with_color(self, png_file_path):
        img_rgb = cv2.imread(png_file_path)
        img_grey = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_adapted = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 101, 9)
        img_stacked = np.repeat(img_adapted[...,None],3,axis=2)
        resized = cv2.resize(img_stacked, (200,200), interpolation=cv2.INTER_AREA)
        bg_img = 255 * np.ones(shape=(256,256,3))
        bg_img[27:227, 27:227,:] = resized
        bg_img /= 255
        return bg_img
    
    def resize_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        img = img.astype('float32')
        img /= 255
        return img
    
    def resize_img1(self, img_path):
        img = cv2.imread(img_path)
        resized = cv2.resize(img, (200,200), interpolation=cv2.INTER_CUBIC)
        img = 255 * np.ones(shape=(256,256,3))
        img[27:227, 27:227,:] = resized
        img = img.astype('float32')
        img /= 255
        return img
    
    def save_resized_img_arrays(self, resized_img_arrays, sample_ids, output_folder):
        count = 0
        for img_arr, sample_id in zip(resized_img_arrays, sample_ids):
            npz_filename = "{}/{}.npz".format(output_folder, sample_id)
            np.savez_compressed(npz_filename, features=img_arr)
            retrieve = np.load(npz_filename)["features"]
            assert np.array_equal(img_arr, retrieve)
            count += 1
        print("Saved down {} resized images to folder {}".format(count, output_folder))