
from tensorflow import keras
import cleverhans.attacks as clhan
from cleverhans.utils_keras import KerasModelWrapper

def fgsm_init(model):
    """
        @brief: Initialize the FGSM module with the Keras Model

        @param: Tensor Flow Model

        @return: Intialized FGSM module
    """

    fgsm_wrapper = KerasModelWrapper(model)
    fgsm_session = keras.backend.get_session()

    return clhan.FastGradientMethod(fgsm_wrapper, fgsm_session)

def fgsm_compute_samples(fgsm_, x_real, eps, min_value, max_value, y_tar = None):
    """
        @brief: Compute Adversarial examples using the Fast Gradient Sign Method 

        @param: FGMS module, true features vector, epsilon, minimum value of the feature, maximum value of the feature, 
                target class (one hot encoded) if any
        
        @return: Adversarial Examples
    """
    
    fgsm_params = {
            'eps' : eps,
            'clip_min' : min_value,
            'clip_max' : max_value,
            'y_target' : y_tar
            }

    return fgsm_.generate_np(x_real, **fgsm_params)


def basic_iter_init(model):
    """
        @brief: Initialize the Basic Iterative module with Keras model
        @param: Keras model (TensorFlow)
        @return: Intialized Basic Iterative module
    """

    biter_wrapper = KerasModelWrapper(model)
    biter_session = keras.backend.get_session()

    return clhan.BasicIterativeMethod(biter_wrapper, biter_session)

def basic_iter_compute_samples(biter_, x_real, eps, min_value, max_value, n_iter = 10, y_tar = None):
    """
        @brief: Compute Adversarial examples using the Basic Iterative Method
        @param: BITER module, true feature vector, epsilon, minimum value of the feature, maximum value of the 
                feature, number of iterations, and target class (one-hot encoded) if any
        @return: Adversarial Examples
    """

    biter_params = {
            'eps_iter' : eps / n_iter,
            'eps' : eps,
            'nb_iter' : n_iter,
            'y_target' : y_tar,
            'clip_min' : min_value,
            'clip_max' : max_value
            }

    return biter_.generate_np(x_real, **biter_params)


def adversarial_evaluate_model(model, X, Y, target_class = None):
    keras.backend.learning_phase()
    # The learning phase flag is a bool tensor to be passed as input to any Keras function that uses a different behaviour at train
    # time and test time
    
    # An internal error was raised with Basic Iterative Method when using Dropout layers in the model. This was solved with learning
    # phase set to 0 i.e. test time. A similar issues has already been raised on GitHub. https://keras.io/backend/#learning_phase
    keras.backend.set_learning_phase(0)

    # Min and Max value in the data
    try:
        min_value = min(min(X.values.flatten()), min(X.values.flatten()))
        max_value = max(max(X.values.flatten()), max(X.values.flatten()))
    except:
        min_value = min(min(X.flatten()), min(X.flatten()))
        max_value = max(max(X.flatten()), max(X.flatten()))
    
    # Epsilon values and number of iterations
    epsilons = [0.01, 0.05, 0.1, 0.5, 1]
    n_iterations = 10
    
    # FGSM
    fgsm_results = {}
    fgsm_model = fgsm_init(model)
    
    # Basic Iterative
    biter_results = {}
    biter_model = basic_iter_init(model)
    
    # Compute the adversarial examples and then evaluate the model on the 
    # computed examples
    for eps in epsilons:
        print("For epsilon {}".format(eps))
        
        # FGSM
        adv_examples = fgsm_compute_samples(fgsm_model, X, eps, min_value, max_value, target_class)
        fgsm_results[eps] = model.evaluate(adv_examples, Y)
        
        # Basic Iterative
        adv_examples = basic_iter_compute_samples(biter_model, X, eps, min_value, max_value, n_iterations, target_class)
        biter_results[eps] = model.evaluate(adv_examples, Y)
        
    return fgsm_results, biter_results

    