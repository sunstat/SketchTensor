import tensorly as tl
from util import randomMatrixGenerator
import numpy as np
from operator import mul

class Skech(object):

    def __init__(self, X, k, s, pass_type = 1, typ = 'g', random_seed, sparse_factor = 0.1):
        '''
        :param X: tensor being skeched
        :param k:
        :param s: s>k
        :param random_seed: random_seed
        :param sparse_factor: only typ == 'sp', p matters representing the sparse factor
        '''
        tl.set_backend('numpy')
        self.X = X
        self.N = len(X.shape)
        self.sketchs = []
        np.random.seed(self.random_seed)
        for n in range(self.N):
            n1 = X.shape[n]
            n2 = np.size(X)/n1
            rm = randomMatrixGenerator(n2, k, 1, typ, self.random_seed[n], sparse_factor)
            self.sketchs.append(np.dot(tl.unfold(X, mode=n), rm))
        if pass_type == 1:

            tl.tenalg.mode_dot(X, M, mode=1)


    def get_sketchs(self):
        return self.sketchs, self.rand_seeds


