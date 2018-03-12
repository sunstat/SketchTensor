import numpy as np
from scipy import fftpack
import tensorly as tl

def randomMatrixGenerator(m, n, std = 1, typ ='g', rand_seed = None, sparse_factor = 0.1):
    '''
    :param m: # of rows in the matrix
    :param n: # of columns in the matrix
    :param std: standard deviation
    :param typ: 'g' for gauss, 'u' for uniform, 's' for Scrambled SRFT; 'sp' 
    for Sparse Sign
    :return: the generated matrix
    '''
    if rand_seed:
        np.random.seed(rand_seed)

    types = set(['g', 'u', 'sp', 's'])
    assert typ in types, "please aset your type of random variable correctly"

    if typ == 'g':
        return np.random.normal(size = (m,n))*std
    elif typ == 'u':
        return np.random.uniform(low = -1, high = 1, size = (m,n))*np.sqrt(3)*std
    elif typ == 'sp':
        return np.random.binomial(n = 1,p = sparse_factor,size = (m,n))*np.random.uniform(low = -1, high = 1, size = (m,n))*np.sqrt(3)*std
    else:
        # Scramble SSRFT map: R*F*PI*F*PI', where R: random uniform with std,
        # F: Discrete cousine transform; PI: Signed permutation matrix
        i = np.identity(n)
        rr = np.range(n)
        np.random.shuffle(rr)
        perm = np.take(i, rr, axis=0)
        return (np.random.uniform(low = -1, high = 1, size = (m,n))).dot(fftpack.dct(perm)).dot(fftpack.dct(perm.T))

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
