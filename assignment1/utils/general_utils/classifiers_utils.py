import numpy as np
import sys
sys.setrecursionlimit(100000)


def mode(vector):
    """
    Choose most popular element of vector.
    Break ties by choosing the smaller label
    :param vector: numpy array: input array, which could be converted into "int32" type
    :return: numpy element: of most popular element
    """
    result = np.argmax(np.bincount(np.array(vector, "int32")))
    return result


def recursive_computation(A, B, n):
    """
    Compute l1 (manhattan) distance from each row matrix B to each row of matrix A. \n
    So that [i, j] elements is the distance between i-th row of B and j-th row of A
    :param A: numpy array: (n, D) - size of matrix
    :param B: numpy array: (l, D) - size of matrix
    :param n: integer: number of columns, which are used to compute distance
    :return: numpy array: (l, n) - size of matrix
    """
    if n == 0:
        s = np.abs(A[:, 0, None] - B[:, 0])
    else:
        s = recursive_computation(A, B, n-1) + np.abs(A[:, n, None] - B[:, n])
    return s
