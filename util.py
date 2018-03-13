import numpy as np
from scipy import fftpack
import tensorly as tl


class RandomInfoBucket(object):

    def __init__(self, std=1, typ='g', random_seed = None, sparse_factor = 0.1):
        self.std = std
        self.typ = typ
        self.random_seed = random_seed
        self.sparse_factor = sparse_factor

    def get_info(self):
        return self.std, self.typ, self.random_seed, self.sparse_factor

def random_matrix_generator(m, n, info_bucket):

    std, typ, random_seed, sparse_factor = info_bucket.get_info()
    if random_seed:
        np.random.seed(random_seed)

    types = set(['g', 'u', 'sp', 's'])
    assert typ in types, "please aset your type of random variable correctly"

    if typ == 'g':
        return np.random.normal(size = (m,n))*std
    elif typ == 'u':
        return np.random.uniform(low = -1, high = 1, size = (m,n))*np.sqrt(3)*std
    elif typ == 'sp':
        return np.random.binomial(n = 1,p = sparse_factor,size = (m,n))*np.random.uniform(low = -1, high = 1, size = (m,n))*np.sqrt(3)*std


def tensorGenHelp(core,arms):
    ''' 
    :param size: array. the length of the tensor
    :param rk: array. The tucker rank 
    ''' 
    for i in np.arange(length(arms)): 
        prod = tl.tenalg.mode_dot(core,arms[i],mode =i)
    return prod 

def mse(): 
    pass


if __name__ == "__main__":
    def f():
        np.random.seed(1)
        print(randomMatrixGenerator(3, 3, std=1, typ='g', rand_seed=None, sparse_factor=0.1))
        print(randomMatrixGenerator(3, 3, std=1, typ='g', rand_seed=None, sparse_factor=0.1))

    f()
    f()

