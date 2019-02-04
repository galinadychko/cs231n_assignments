import time
import numpy as np


def accuracy(y_test_pred, y_test_true):
    """
    Count the ratio of correct classified objects
    :param y_test_pred: np.array; predicted labels
    :param y_test_true: np.array; true labels
    :return: float; ratio of correct predictions
    """
    num_test = len(y_test_pred)
    num_correct = np.sum(y_test_pred == y_test_true)
    accuracy_value = float(num_correct) / num_test
    return accuracy_value


def time_performance(f, *args):
    """
    Note the function f with the parameters *args execution time
    :param f: function;
    :param args: function f arguments;
    :return: float; execution time
    """
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic
