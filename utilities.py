
import numpy as np

def get_one_hot_target(target, n_classes):
    """
        @brief: Get One-Hot encoded arary at position target with size 1 * n_size

        @param: Target index and the size 

        @return: An array with all zeros except a 1 at index target

    """
    if target >= n_classes:
        print("Invalid index value for one-hot encoding")

    ret = np.zeros((1, n_classes), dtype=int)
    ret[0][target] = 1
    return ret
	
def get_target_score(target, predictions):
    '''
        @brief: Get the percentage of target in predictions
        
        @param: Target value and prediction list
        
        @return: Percentage
    '''
    total = len(predictions)
    count = 0
    for x in predictions:
        if x == target:
            count += 1
            
    return (count / total)
	
def show_adversarial_results(data):
    for k, v in zip(data.keys(), data.values()):
        print("Epsilon = {} \t Loss = {:.3f} \t Accuracy = {:.3f}".format(k, v[0], v[1] * 100))