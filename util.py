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


def tensor_gen_help(core,arms):
    '''
    :param core: the core tensor in higher order svd s*s*...*s
    :param arms: those arms n*s
    :return:
    '''
    for i in np.arange(len(arms)):
        prod = tl.tenalg.mode_dot(core,arms[i],mode =i)
    return prod 


def square_tensor_gen(n, r, dim = 3, typ = 'id', noise_level = 0):
    '''
    :param n: size of the tensor generated n*n*...*n
    :param r: rank of the tensor or equivalently, the size of core tensor
    :param dim: # of dimensions of the tensor, default set as 3
    :param typ: identity as core tensor or low rank as core tensor
    :param noise_level:
    :return:
    '''
    types = set(['id', 'lk'])
    total_num = np.power(n, dim)
    assert typ in types, "please set your type of tensor correctly"
    if typ == 'id':
        identity = np.zeros(np.repeat(n, dim))
        for i in np.arange(min(r)):
            identity[(np.repeat(i,dim))] = 1
        noise = np.random.normal(0,1,np.repeat(n, dim))
        return identity+noise*np.sqrt(noise_level*r/np.product(total_num))

    if typ == "lk":
        core = np.random.uniform(0,1,np.repeat(n, dim))
        arms = []
        for i in np.arange(len(size)):
            arm = np.random.normal(0,1, size = (n,r))
            arm, _ = np.linalg.qr(arm)
            arms.append(arm)
            core = tl.tenalg.mode_dot(core, arm, mode=i)
        return core
