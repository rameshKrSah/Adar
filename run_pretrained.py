# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:28:40 2019

@author: rames
"""

from __future__ import print_function
from __future__ import division

import numpy as np
from tensorflow import keras
import adversarial as adv
import utilities as utl
import data_processing as data


if __name__ == '__main__':
    # ============================= EXPERIMENTS WITH DNN MODEL ================================================= #
	# Load the DNN model and data
    print("Loading the processed feature data")
    f_x_train, f_y_train, f_x_test, f_y_test = data.load_saved_feature_data()
    
    print("Loading the pretrained DNN model")
    dnn_model = data.load_saved_dnn_model()
	
	# Evaluate the model on the test and train data
    l, a = dnn_model.evaluate(f_x_train, f_y_train)
    print("DNN model on train data, loss {:.3f}, accuracy {:.3f}".format(l, a * 100))
    l, a = dnn_model.evaluate(f_x_test, f_y_test)
    print("DNN model on test data, loss {:.3f}, accuracy {:.3f}".format(l, a * 100))
    	
    # Compute untargeted adversarial examples using the DNN model for different 
    # values of epsilon with different methods and plots some results.
    u_dnn_fgsm_results, u_dnn_biter_results = adv.adversarial_evaluate_model(dnn_model, f_x_test, f_y_test)
    
    print("============= Results of Untargeted attack on the DNN model =============")
    print("Results shows the performance of the DNN model on the adversarial examples computed using FGSM method at different values of epsilon")
    utl.show_adversarial_results(u_dnn_fgsm_results)
    
    print("Results shows the performance of the DNN model on the adversarial examples computed using BITER method at different values of epsilon")
    utl.show_adversarial_results(u_dnn_biter_results)
    
    # Compute targeted adversarial examples using the DNN model
    target_class = 3 # Sit class label
    n_classes = f_y_train.max() + 1
    target_one_hot = utl.get_one_hot_target(target_class, n_classes)
    y_target = np.ones(f_x_test.shape[0]) * target_class
    t_dnn_fgsm_results, t_dnn_biter_results = adv.adversarial_evaluate_model(dnn_model, f_x_test, y_target, target_one_hot)
    
    print("For targeted attack we measure the attack success rate : i.e., the number examples that was classified into the target class to the total number of examples")
    
    print("============= Results of Targeted attack on the DNN model =============")
    print("Results shows the performance (success rate) of the DNN model on the adversarial examples computed using FGSM method at different values of epsilon")
    utl.show_adversarial_results(t_dnn_fgsm_results)
    
    print("Results shows the performance (success rate) of the DNN model on the adversarial examples computed using BITER method at different values of epsilon")
    utl.show_adversarial_results(t_dnn_biter_results)
    
    # ============================= EXPERIMENTS WITH CNN MODEL ================================================= #
    # load the CNN model and data
    print("Loading the preprocessed window data")
    w_x_train, w_y_train, w_x_test, w_y_test = data.load_saved_window_data()
    
    print("Loading the pretrained CNN model")
    cnn_model = data.load_saved_cnn_model()
    	
    n_classes = w_y_train.max() + 1
    w_y_train_one_hot = keras.utils.to_categorical(w_y_train, n_classes)
    w_y_test_one_hot = keras.utils.to_categorical(w_y_test, n_classes)
    
    # Evaluate the CNN model on the train and test data
    l, a = cnn_model.evaluate(w_x_train, w_y_train_one_hot)
    print("CNN model on train data, loss {:.3f}, accuracy {:.3f}".format(l, a * 100))
    l, a = cnn_model.evaluate(w_x_test, w_y_test_one_hot)
    print("CNN model on test data, loss {:.3f}, accuracy {:.3f}".format(l, a * 100))

     # Compute untargeted adversarial examples using the DNN model for different 
    # values of epsilon with different methods and plots some results.
    u_cnn_fgsm_results, u_cnn_biter_results = adv.adversarial_evaluate_model(cnn_model, w_x_test, w_y_test_one_hot)
    
    print("============= Results of Untargeted attack on the CNN model =============")
    print("Results shows the performance of the CNN model on the adversarial examples computed using FGSM method at different values of epsilon")
    utl.show_adversarial_results(u_cnn_fgsm_results)
    
    print("Results shows the performance of the CNN model on the adversarial examples computed using BITER method at different values of epsilon")
    utl.show_adversarial_results(u_cnn_biter_results)
    
    # Compute targeted adversarial examples using the DNN model
    y_target_one_hot = keras.utils.to_categorical(np.ones(w_y_test.shape[0]) * target_class, n_classes)
    t_cnn_fgsm_results, t_cnn_biter_results = adv.adversarial_evaluate_model(cnn_model, w_x_test, y_target_one_hot, y_target_one_hot)
    
    print("============= Results of Targeted attack on the CNN model =============")
    print("Results shows the performance (success rate) of the CNN model on the adversarial examples computed using FGSM method at different values of epsilon")
    utl.show_adversarial_results(t_cnn_fgsm_results)
    
    print("Results shows the performance (success rate) of the CNN model on the adversarial examples computed using BITER method at different values of epsilon")
    utl.show_adversarial_results(t_cnn_biter_results)
    
    
    