import numpy as np
from six.moves import cPickle as pickle
import os

project_path = os.getcwd()


def load_pickle(f):
    """
    Load pickled file f
    :param f: file
    :return: loaded_file
    """
    return pickle.load(f, encoding="latin1")


def load_CIFAR_batch(filename):
    """
    Load pickled data from the filename
    :param filename: string; file name
    :return: (np.array, np.array); (matrix with features, vector with labels)
    """
    with open(filename, "rb") as f:
        data_dictionary = load_pickle(f)
        X = data_dictionary["data"].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.asarray(data_dictionary["labels"], dtype="float")
    return X, Y


def load_CIFAR10(filepath):
    """
    Load pickled dataset CIFAR10, which consists of 5 parts of pickled data (data_batch_) and one test file (/test_batch)
    :param filepath: string; path to the folder with dataset
    :return: (np.array, np.array, np.array, np.array); \
    (train matrix with features, train vector with labels, test matrix with features, test vector with labels)
    """
    all_X = []
    all_Y = []
    for batch_nom in range(1, 6):
        X, Y = load_CIFAR_batch(filepath + "/data_batch_" + str(batch_nom))
        all_X.append(X)
        all_Y.append(Y)

    del X, Y

    X_test, Y_test = load_CIFAR_batch(filepath + "/test_batch")

    return np.concatenate(all_X), np.concatenate(all_Y), X_test, Y_test


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True):
    """
    Read CIFAR10 data from the default directory and split it into train, validation and test parts \
    according the input parameters. Normalize each part, if the corresponding parameter is set.
    :param num_training: int; number of observations in train dataset
    :param num_validation: int; number of observations in validation dataset
    :param num_test: int; number of observations in test dataset
    :param subtract_mean: bool; if normalize datasets according to each mean
    :return:
    """
    X_train, Y_train, X_test, Y_test = load_CIFAR10(project_path+"/cifar-10-batches-py")

    cut_list = list(range(num_training, num_training + num_validation))
    X_val = X_train[cut_list]
    Y_val = Y_train[cut_list]

    cut_list = list(range(num_training))
    X_train = X_train[cut_list]
    Y_train = Y_train[cut_list]

    cut_list = list(range(num_test))
    X_test = X_test[cut_list]
    Y_test = Y_test[cut_list]

    if subtract_mean:
        X_train -= np.mean(X_train, axis=0)
        X_val -= np.mean(X_val, axis=0)
        X_test -= np.mean(X_test, axis=0)

    del cut_list
    return dict(X_train=X_train.transpose(0, 3, 1, 2), Y_train=Y_train,
                X_val=X_val.transpose(0, 3, 1, 2), Y_val=Y_val,
                X_test=X_test.transpose(0, 3, 1, 2), Y_test=Y_test)



