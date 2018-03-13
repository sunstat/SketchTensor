import tensorly as tl
import numpy as np
from operator import mul
from util import random_matrix_generator
from util import RandomInfoBucket




class Sketch(object):

    @staticmethod
    def sketch_random_matrix_generator(tensor_shape, reduced_dim, typ='g', sparse_factor=0.1):
        total_num = np.prod(tensor_shape)
        for n in range(len(tensor_shape)):
            n1 = total_num/tensor_shape[n]
            yield random_matrix_generator(reduced_dim, n1, RandomInfoBucket(std=1, typ=type, sparse_factor = sparse_factor))


    def __init__(self, X, k, random_seed, s = -1, typ = 'g', sparse_factor = 0.1):
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
        self.sketchs = []
        self.random_seed = random_seed
        self.core_sketch = X
        self.tensor_shape = X.shape

        # set the random seed for following procedure
        np.random.seed(random_seed)

        rm_generator = Sketch.sketch_random_matrix_generator(self.tensor_shape, reduced_dim=self.k,
                                                          typ=self.typ, sparse_factor=self.sparse_factor)
        mode_n = 0
        for rm in rm_generator:
            self.sketchs.append(np.dot(tl.unfold(self.X, mode=mode_n), rm))
            mode_n += 1

        if self.s != -1:
            rm_generator = Sketch.sketchRandomMatrixGenerator(self.tensor_shape, reduced_dim=self.s,
                                                              typ=self.typ, sparse_factor=self.sparse_factor)
            mode_n = 0
            for rm in rm_generator:
                self.core_sketch = tl.tenalg.mode_dot(self.core_sketch, rm, mode=mode_n)
                mode_n += 1

    def get_sketchs(self):
        return self.sketchs, self.core_sketch, self.random_seed


