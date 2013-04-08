import time
import platform
import numpy as np


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())
    
def hostname():
    return platform.node()
    
def generate_expid():
    return "%s-%s" % (hostname(), timestamp())


# samplers
def pick_one(*options):
    def sample():
        idx = np.random.randint(len(options))
        return options[idx]
    return sample

def uniform(start, end):
    def sample():
        return np.random.uniform(start, end)
    return sample
    
def uniform_int(start, end): # including 'end'!
    def sample():
        return np.random.randint(start, end + 1)
    return sample
    
def log_uniform(start, end):
    ls, le = np.log(start), np.log(end)
    def sample():
        return np.exp(np.random.uniform(ls, le))
    return sample
    
def log_uniform_int(start, end):
    ls, le = np.log(start), np.log(end + 1)
    def sample():
        return int(np.floor(np.exp(np.random.uniform(ls, le))))
    return sample
