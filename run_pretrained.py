# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:28:40 2019

@author: rames
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import pickle

import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

# Load the datasets
feature_data_file = "resources/UCI_Feature_Data.pickle"
window_data_file = "resources/UCI_Window_Data.pickle"

# First the feature data
file = open(feature_data_file, 'rb')
f_x_train, f_y_train, f_x_test, f_y_test = pickle.load(file)
file.close()

# Second the window data
file = open(window_data_file, 'rb')
w_x_train, w_y_train, w_x_test, w_y_test = pickle.load(file)
file.close()

# Some variables
n_classes = len(np.unique(w_y_train))

# Load the pretrained models
dnn_model_file = "resources/UCI_Feature_Model.h5py"
cnn_model_file = "resources/UCI_Window_Model.h5py"

# First the DNN model trained on feature data
dnn_model = keras.models.load_model(dnn_model_file)

# Evaluate the model on the test and train data
l, a = dnn_model.evaluate(f_x_train, f_y_train)
print("DNN model on train data, loss {:.3f}, accuracy {:.3f}".format(l, a * 100))
l, a = dnn_model.evaluate(f_x_test, f_y_test)
print("DNN model on test data, loss {:.3f}, accuracy {:.3f}".format(l, a * 100))


# Second the CNN model trained on the window data
cnn_model = keras.models.load_model(cnn_model_file)
w_y_train_one_hot = keras.utils.to_categorical(w_y_train, n_classes)
w_y_test_one_hot = keras.utils.to_categorical(w_y_test, n_classes)

# Evaluate the CNN model on the train and test data
l, a = cnn_model.evaluate(w_x_train, w_y_train_one_hot)
print("CNN model on train data, loss {:.3f}, accuracy {:.3f}".format(l, a * 100))
l, a = cnn_model.evaluate(w_x_test, w_y_test_one_hot)
print("CNN model on test data, loss {:.3f}, accuracy {:.3f}".format(l, a * 100))





