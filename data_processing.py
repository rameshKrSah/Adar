# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:20:27 2019

@author: ramesh
"""

'''
 We assume that the UCI data is available in the folder named "data" with the 
 following file structure.
 
 -data
     -UCI
         -train
         -test
'''

import os
import pickle
import numpy as np
import pandas as pd

data_folder = "data/UCI/"

def load_feature_data():
    feature_file = data_folder + "features.txt"
    f_train = data_folder + "train/X_train.txt"
    f_test = data_folder + "test/X_test.txt"
    l_train = data_folder + "train/y_train.txt"
    l_test = data_folder + "test/y_test.txt"
    
    # Load the feature names
    file = open(feature_file, 'r')
    features = file.read().splitlines()
    features = map(lambda x: x.rstrip().lstrip().split(), features)
    features = [list(map(str, line))[1] for line in features]
    
    # Load the train features file
    file = open(f_train, 'r')
    features_train = file.read().splitlines()
    features_train = map(lambda x: x.rstrip().lstrip().split(), features_train)
    features_train = [list(map(float, line)) for line in features_train]
    
    # Load the test features file
    file = open(f_test, 'r')
    features_test = file.read().splitlines()
    features_test = map(lambda x: x.rstrip().lstrip().split(), features_test)
    features_test = [list(map(float, line)) for line in features_test]
    
    # Load the train labels
    file = open(l_train, 'r')
    train_label = file.read().splitlines()
    train_label = [int(line) for line in train_label]
    y_train = np.array(train_label)
    y_train = y_train - 1
    
    # Load the test labels
    file = open(l_test, 'r')
    test_label = file.read().splitlines()
    test_label = [int(line) for line in test_label]    
    y_test = np.array(test_label)
    y_test = y_test - 1
    
    # Save the data
    x_train = pd.DataFrame(features_train, columns=features)
    x_test = pd.DataFrame(features_test, columns=features)

    file = open("resources/UCI_Feature_Data.pickle", "wb")
    pickle.dump([x_train, y_train, x_test, y_test], file)
    file.close()
    
def read_data_from_file(file):
    with open(file, 'r') as file:
        data = file.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
    return data

def load_window_data():
    train_data_folder = data_folder + "train/Inertial Signals/"
    test_data_folder = data_folder + "test/Inertial Signals/"
    train_label_file = data_folder + "train/y_train.txt"
    test_label_file = data_folder + "test/y_test.txt"
    
    train_data = []
    test_data = []
    
    for file in os.listdir(train_data_folder):       
        data = read_data_from_file(train_data_folder+file)
        train_data.append(data)
    
    # changing the dimensions from (9, 7352, 128) i.e. 9 different channels, with 7452 rows and 128 columns to (7352, 128, 9)
    train_data = np.array(train_data).transpose(1, 2, 0)
    
    for file in os.listdir(test_data_folder):
        data = read_data_from_file(test_data_folder+file)
        test_data.append(data)
    
    # changing the dimensions from (9, 7352, 128) i.e. 9 different channels, with 7452 rows and 128 columns to (7352, 128, 9)
    test_data = np.array(test_data).transpose(1, 2, 0)
    
    # Load the train labels
    file = open(train_label_file, 'r')
    train_label = file.read().splitlines()
    train_label = [int(line) for line in train_label]
    y_train = np.array(train_label)
    y_train = y_train - 1
    
    # Load the test labels
    file = open(test_label_file, 'r')
    test_label = file.read().splitlines()
    test_label = [int(line) for line in test_label]    
    y_test = np.array(test_label)
    y_test = y_test - 1
    
    # Save the window data as a pickle
    file = open("resources/UCI_Window_Data.pickle", "wb")
    pickle.dump([train_data, y_train, test_data, y_test], file)
    file.close()
    
    
if __name__ == '__main__':
    # Read the feature data, process it into proper format, and save it as a pickle
    load_feature_data()
    
    # Read the window data, process it into proper format, and save it as a pickle
    load_window_data()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    