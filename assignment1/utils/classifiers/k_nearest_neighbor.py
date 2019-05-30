import numpy as np
from utils.classifiers.export_modules import *
from utils.general_utils.classifiers_utils import *

#####################################################################
# TODO:                                                             #
# L1 distance                                                       #
#####################################################################


class KNearestNeighbor:
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X_train, Y_train):
        """
        Memorize the trainig data \n
        (matrix of objects features and their labels separately)
        :param X_train: numpy array: (Ntrain, D) - size of matrix with Ntrain objects and D features
        :param Y_train: numpy array: (Ntrain, ) - size of matrix with Ntrain labels of train data, where \n
             y[i] is the label for X[i]
        :return: None
        """
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test, k, num_loops=0, distance="L2"):
        """
        Make label prediction for each object from the input data
        :param X_test: numpy array: (N, ) - size of matrix with Ntest objects and D features
        :param k: integer: number of neighbors, which labels are used for making prediction
        :param num_loops: numeric: determine how many loops should be used for computing distance \n
        (0, 1, 2 are available)
        :return: numeric: label of each object from the input data. \n
        y[i] is the predicted label for the test point X[i]
        """
        if num_loops not in [0, 1, 2]:
            print("Not correct number of loops: %d" % num_loops)
        elif distance != "L1":
            dist = getattr(self, 'compute_dist_' + str(num_loops) + "_loop")(X_test)
        elif distance == "L1":
            dist = self.compute_dist_l1(X_test)
        return self.predict_labels(dist, k)

    def compute_dist_l1(self, X_test):
        """
        Compute the distance between each test point in X and each training point \n
        in self.X_train.
        L1 distance is used.
        :param X_test: numpy array: (Ntest, D) - size of matrix containing test data
        :return: numpy ndarray: (Ntest, Ntrain) - size of distances, where dist[i, j] contains \n
        distance between i-th test object and j-th train object
        """
        n = X_test.shape[1]
        dist = recursive_computation(X_test, self.X_train, (n-1))
        return dist

    def compute_dist_2_loop(self, X_test):
        """
        Compute the distance between each test point in X and each training point \n
        in self.X_train using a nested loop over both the training data and the \n
        test data. \n
        L2 (Euclidean) distance is used.
        :param X_test: numpy array: (Ntest, D) - size of matrix containing test data
        :return: numpy ndarray: (Ntest, Ntrain) - size of distances, where dist[i, j] contains \n
        distance between i-th test object and j-th train object
        """
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dist = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                dist[i, j] = np.sqrt(np.sum((X_test[i, :] - self.X_train[j, :])**2))
        return dist

    def compute_dist_1_loop(self, X_test):
        """
        CCompute the distance between each test point in X and each training point \n
        in self.X_train using a single loop over the test data. \n
        L2 (Euclidean) distance is used.
        :param X_test: numpy array: (N, ) - size of matrix containing test data
        :return: numpy ndarray: (Ntest, Ntrain) - size of distances, where dist[i, j] contains \n
        distance between i-th test object and j-th train object
        """
        num_test = X_test.shape[0]
        dist = np.zeros((num_test, self.X_train.shape[0]))
        for i in range(num_test):
            dist[i, :] = np.sqrt(np.sum((self.X_train - X_test[i, :])**2, axis=1))
        return dist

    def compute_dist_0_loop(self, X_test):
        """
        Compute the distance between each test point in X and each training point \n
        in self.X_train using no explicit loops. \
        L2 (Euclidean) distance is used.
        :param X_test: numpy array: (N, ) - size
        :return: numpy ndarray: (Ntest, Ntrain) - size of distances, where dist[i, j] contains \n
        distance between i-th test object and j-th train object
        """
        num_test, num_train = X_test.shape[0], self.X_train.shape[0]

        dist = np.sqrt(np.sum(X_test ** 2, axis=1).reshape(num_test, 1) + \
                       np.sum(self.X_train ** 2, axis=1) - \
                       2*X_test.dot(self.X_train.T))
        return dist

    def predict_labels(self, dist, k):
        """
        Given a matrix of distances between test points and training points, \n
        predict a label for each test point. \n
        Break ties by choosing the smaller label.
        :param dist: numpy array: (Ntest, Ntrain) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point
        :param k: integer: number of nearest neighbors for choosing labels
        :return: numpy ndarray: (Ntest, ) of labels for test data
        """
        num_test, num_train = dist.shape
        label_indexes = np.argsort(dist, axis=1)[:, :k]
        ytrain_repeaed = np.tile(self.Y_train, (num_test, 1))
        k_labels = np.take(ytrain_repeaed, label_indexes)
        predicted_label = np.apply_along_axis(mode, 1, k_labels)
        return predicted_label
