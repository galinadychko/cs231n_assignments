import numpy as np


#####################################################################
# TODO:                                                             #
# Write documentation                                               #
#####################################################################

def mode(vector):
    result = np.argmax(np.bincount(np.array(vector, "int32")))
    return result
