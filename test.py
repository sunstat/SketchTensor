from util import randomMatrixGenerator
import numpy as np

def f():
    np.random.seed(1)
    print(randomMatrixGenerator(3, 3, std=1, typ='g', rand_seed=None, sparse_factor=0.1))
    print(randomMatrixGenerator(3, 3, std=1, typ='g', rand_seed=None, sparse_factor=0.1))


f()
f()