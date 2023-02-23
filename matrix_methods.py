import numpy as np
import numba as nb

"""
Todo:
    Vectorize batch generation
"""

def generate_walk(length):
    """
    Generates a sequence representing a random walk of a prescribed length.
    """
    rng = np.random.default_rng()
    assert length > 0
    return rng.choice([True, False], size=length)

def convert_walk_to_matrix(walk):
    """
    Computes the matrix in SL(2,Z) corresponding to the end of a random walk.
    """
    matrices = np.zeros(shape = (len(walk), 2, 2))
    matrices[walk] = np.array([[0,-1],[1,0]])
    matrices[~walk] = np.array([[0,-1],[1,1]])
    if matrices.shape[0] == 1:
        return matrices[0]
    else:
        return multi_matmul(matrices)

@nb.jit(nopython=True)
def multi_matmul(matrices):
    """
    Just-in-time multiplication of many matrices.
    Faster than np.linalg.multi_dot when multiplying a large number of small matrices.
    """
    prod = np.identity(matrices.shape[2])
    for i in nb.prange(matrices.shape[0]):
        temp = matrices[i]
        prod = np.dot(prod, temp)
    return prod

def create_random_sl2z(length):
    if length == 0:
        return np.identity(2)
    return convert_walk_to_matrix(generate_walk(length))

def generate_multi_sl2z(max_length, num_matrices):
    """
    Generate num_matrices number of random elements of SL(2,Z).
    Random matrices are generated via random walk of length <= max_length.
    """
    rng = np.random.default_rng()
    # randomly generate list of random walk lengths
    lengths = rng.integers(low = 0, high = max_length, size=num_matrices, endpoint=True)
    batch = np.zeros(shape = (num_matrices, 2, 2))
    for idx, length in enumerate(lengths):
        batch[idx] = create_random_sl2z(length)
    return batch.astype(int)

def det_multi(matrices):
    """
    Compute elementwise determinants of an array of 2x2 matrices.
    """
    return matrices[:,0,0] * matrices[:,1,1] - matrices[:,0,1] * matrices[:,1,0]