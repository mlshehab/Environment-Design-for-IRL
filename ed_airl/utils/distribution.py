import numpy as np


def loguniform(a, b, size=None):
    return np.exp(np.random.uniform(np.log(a), np.log(b), size))
    
    
    
