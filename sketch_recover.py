import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from util import RandomInfoBucket
from util import random_matrix_generator

class SketchTwoPassRecover(object):
    def __init__(self, X, sketchs, rank):
        tl.set_backend('numpy')
        self.arms = []
        self.core_tensor = None
        self.X = X
        self.sketchs = sketchs
        self.rank = rank

    def recover(self):
        # get orthogonal basis for each arm
        Qs = []
        for sketch in self.sketchs:
            Q, _ = np.linalg.qr(sketch)
            Qs.append(Q)

        #get the core_(smaller) to implement tucker
        core_tensor = self.X
        N = len(self.X.shape)
        for mode_n in range(N):
            Q = Qs[n]
            core_tensor = tl.tenalg.mode_dot(core_tensor, Q.T, mode=mode_n)
        core_tensor, factors = tucker(core_tensor, ranks=np.repeat(self.rank, N))
        self.core_tensor = core_tensor

        #arm[n] = Q.T*factors[n]
        for n in range(len(factors)):
            self.arms.append(np.dot(Qs[n], factors[n]))
        X_hat = self.core_tensor
        for n in range(len(factors)):
            X_hat = tl.tenalg.mode_dot(core_tensor, self.arms[n], mode=n)
        error = self.X-X_hat
        error = np.norm(error.reshape(np.size(error),1), 'fro')
        mse = error/np.size(self.X)
        return self.arms, self.core_tensor, error, mse

class SketchOnePassRecover(object):

    @staticmethod
    def get_phis(info_bucket, tensor_shape, k, s):
        total_num = np.prod(tensor_shape)
        # generate omegas which we do not need store
        for i in range(len(tensor_shape)):
            n1 = tensor_shape[i]
            n1 = total_num/n1
            _ = random_matrix_generator(n1, k, info_bucket)



    def __init__(self, sketchs, core_sketch, rank):
        tl.set_backend('numpy')
        self.arms = []
        self.core_tensor = None
        self.sketchs = sketchs
        self.core_sketch = core_sketch
        self.rank = rank

    def recover(self):
        Qs = []
        for sketch in self.sketchs:
            Q, _ = np.linalg.qr(sketch)
            Qs.append(Q)

        core_tensor = self.core_sketch
        N = len(self.core_sketch.shape)
        for n in range(N):
            tl.tenalg.mode_dot(core_tensor, , mode=n)










