import numpy as np
from utils.classifiers.export_modules import *
from utils.general_utils.classifiers_utils import *

#####################################################################
# TODO:                                                             #
# Write documentation                                               #
#####################################################################


class KNearestNeighbor:
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X_train, Y_train):
        """
        Memorize the trainig data \
        (matrix of objects features and their labels separately)
        :param X_train: numpy array: (N, D) - size
        :param Y_train: numpy array: (N, ) - size
        :return: None
        """
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test, k, num_loops=0):
        """
        Make label prediction for each object from the input data
        :param X_test: numpy array: (N, ) - size
        :param k: numeric: number of neighbors, which labels are used for making prediction
        :param num_loops: numeric: determine how many loops should be used for computing distance
        :return: numeric: label of each object from the input data
        """
        if num_loops not in [0, 1, 2]:
            print("Not correct number of loops: %d" % num_loops)
        else:
            dist = getattr(self, 'compute_dist_' + str(num_loops) + "_loop")(X_test)
        return self.predict_labels(dist, k)

    def compute_dist_2_loop(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dist = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                dist[i, j] = np.sqrt(np.sum((X_test[i, :] - self.X_train[j, :])**2))
        return dist

    def compute_dist_1_loop(self, X_test):
        num_test = X_test.shape[0]
        dist = np.zeros((num_test, self.X_train.shape[0]))
        for i in range(num_test):
            dist[i, :] = np.sqrt(np.sum((self.X_train - X_test[i, :])**2, axis=1))
        return dist

    def compute_dist_0_loop(self, X_test):
        num_test, num_train = X_test.shape[0], self.X_train.shape[0]

        dist = np.sqrt(np.sum(X_test ** 2, axis=1).reshape(num_test, 1) + \
                       np.sum(self.X_train ** 2, axis=1) - \
                       2*X_test.dot(self.X_train.T))
        return dist

    def predict_labels(self, dist, k):
        num_test, num_train = dist.shape
        label_indexes = np.argsort(dist, axis=1)[:, :k]
        ytrain_repeaed = np.tile(self.Y_train, (num_test, 1))
        k_labels = np.take(ytrain_repeaed, label_indexes)
        predicted_label = np.apply_along_axis(mode, 1, k_labels)
        return predicted_label
