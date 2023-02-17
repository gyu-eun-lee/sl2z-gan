import numpy as np
from numba import jit

"""
Todo:
    Vectorize batch generation
"""

def generate_walk(length):
    """
    Generates a sequence representing a random walk of a prescribed length.
    """
    assert length > 0
    return np.random.choice([True, False], size=length)

def convert_walk_to_matrix(walk):
    """
    Computes the matrix in SL(2,Z) corresponding to the end of a random walk.
    """
    matrices = np.zeros(shape = (len(walk), 2, 2))
    matrices[walk] = np.array([[0,-1],[1,0]])
    matrices[~walk] = np.array([[0,-1],[1,1]])
    if matrices.shape[0] == 1:
        return matrices[0].astype(int)
    else:
        return np.linalg.multi_dot(matrices).astype(int)

@jit(forceobj=True)
def create_random_sl2z(length):
    if length == 0:
        return np.identity(2)
    return convert_walk_to_matrix(generate_walk(length))

def create_batch_sl2z(max_length, batch_size):
    lengths = np.random.randint(low = 0, high = max_length, size=batch_size)
    batch = np.zeros(shape = (batch_size, 2, 2))
    for idx, length in enumerate(lengths):
        batch[idx] = create_random_sl2z(length)
    return batch

def det_batch(batch):
    return batch[:,0,0] * batch[:,1,1] - batch[:,0,1] * batch[:,1,0]