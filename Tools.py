import numpy as np
import os
def checkFile(path):
    '''
    if filepath not exist make it
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def standard(x):
    max_value = np.max(x)
    min_value = np.min(x)
    if max_value == min_value:
        return np.zeros_like(x)
    return (x - min_value) / (max_value - min_value)