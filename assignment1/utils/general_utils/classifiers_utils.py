import numpy as np


def mode(vector):
    result = np.argmax(np.bincount(np.array(vector, "int32")))
    return result
