import tensorly as tl
from util import randomMatrixGenerator
import numpy as np
from operator import mul

class Skechy(object):

    def __init__(self, X, k, s, typ = 'g', rand_seeds = None, sparse_factor = 0.1):
        '''
        :param X: tensor being skeched
        :param k:
        :param s: s>k
        :param rand_seeds: array of rand seeds
        :param sparse_factor: only typ == 'sp', p matters representing the sparse factor
        '''
        tl.set_backend('numpy')
        self.X = X
        self.N = len(X.shape)
        self.sketchy_matrices = []
        assert self.N == len(rand_seeds), 'random seeds size does not match with tensor size'
        for n in range(self.N):
            n1 = X.shape[n]
            n2 = np.size(X)/n2
            rm = randomMatrixGenerator(n1, n2, 1, typ, rand_seeds[n], sparse_factor)
            self.sketchy_matrices.append(np.dot(tl.unfold(X, mode=n), rm))


    def get_sketchy(self):
        return self.sketchy_matrices, self.rand_seeds

if __name__ == '__main__':

    print(Skechy(np.random.normal(size = (10,15,20)),5,8,rand_seeds = (1,2,3)))


