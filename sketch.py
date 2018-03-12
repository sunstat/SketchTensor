import tensorly as tl
from util import randomMatrixGenerator
import numpy as np
from operator import mul

class Skech(object):

    @staticmethod
    def sketchRandomMatrixGenerator(tensor_shape, reduced_dim, typ='g', sparse_factor=0.1):
        total_num = np.prod(tensor_shape)
        for n in range(len(tensor_shape)):
            n1 = total_num/tensor_shape[n]
            yield randomMatrixGenerator(reduced_dim, n1, 1, typ=typ, sparse_factor=sparse_factor)


    def __init__(self, X, k, random_seed, s = -1, pass_type = 1, typ = 'g', sparse_factor = 0.1):
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
        self.s = s
        self.k = k
        self.typ = typ
        self.sparse_factor = sparse_factor
        self.pass_type = pass_type
        self.sketchs = []
        self.random_seed = random_seed
        self.core_sketch = X
        self.tensor_shape = X.shape
        rm_generator = Skech.sketchRandomMatrixGenerator(self.tensor_shape, reduced_dim=self.k, typ='g', sparse_factor=0.1)
        mode_n = 0
        for rm in rm_generator:
            self.sketchs.append(np.dot(tl.unfold(X, mode=mode_n), rm))
            mode_n+=1
        rm_generator = Skech.sketchRandomMatrixGenerator(self.tensor_shape, reduced_dim=self.s, typ='g', sparse_factor=0.1)
        mode_n = 0
        for rm in rm_generator:
            self.sketchs.append(np.dot(tl.unfold(X, mode=mode_n), rm))
            mode_n+=1



    def get_sketchs(self):
        return self.sketchs, self.random_seed


