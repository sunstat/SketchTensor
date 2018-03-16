import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from util import RandomInfoBucket
from util import random_matrix_generator
from util import generate_super_diagonal_tensor

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
            Q = Qs[mode_n]
            core_tensor = tl.tenalg.mode_dot(core_tensor, Q.T, mode=mode_n)
        core_tensor, factors = tucker(core_tensor, ranks=[self.rank for _ in range(N)])
        self.core_tensor = core_tensor

        #arm[n] = Q.T*factors[n]
        for n in range(len(factors)):
            self.arms.append(np.dot(Qs[n], factors[n]))
        X_hat = tl.tucker_to_tensor(self.core_tensor, self.arms)
        error = self.X-X_hat
        error = np.linalg.norm(error.reshape(np.size(error),1), 'fro')
        mse = error/np.size(self.X)
        return self.arms, self.core_tensor, error, mse

class SketchOnePassRecover(object):

    @staticmethod
    def get_phis(Rinfo_bucket, tensor_shape, k, s):
        total_num = np.prod(tensor_shape)
        phis = []
        # generate omegas which we do not need store
        for i in range(len(tensor_shape)):
            n1 = tensor_shape[i]
            n1 = total_num//n1
            _ = random_matrix_generator(n1, k, Rinfo_bucket)

        for i in range(len(tensor_shape)):
            phis.append(random_matrix_generator(s, tensor_shape[i], Rinfo_bucket))

        return phis

    def __init__(self, sketchs, core_sketch, Tinfo_bucket, Rinfo_bucket, X):
        tl.set_backend('numpy')
        self.arms = []
        self.core_tensor = None
        self.sketchs = sketchs
        self.tensor_shape, self.k, self.rank, self.s = Tinfo_bucket.get_info()
        self.Rinfor_bucket = Rinfo_bucket
        self.core_sketch = core_sketch
        self.X = X


    def recover(self):
        Qs = []
        for sketch in self.sketchs:
            Q, _ = np.linalg.qr(sketch)
            Qs.append(Q)
        phis = SketchOnePassRecover.get_phis(self.Rinfor_bucket, tensor_shape = self.tensor_shape, k = self.k, s = self.s)
        self.core_tensor = self.core_sketch
        dim = len(self.tensor_shape)
        for mode_n in range(dim):
            self.core_tensor = tl.tenalg.mode_dot(self.core_tensor, np.linalg.pinv(np.dot(phis[mode_n], Qs[mode_n])), mode=mode_n)

        core_tensor, factors = tucker(self.core_tensor, ranks=[self.rank for _ in range(dim)])
        self.core_tensor = core_tensor
        for n in range(dim):
            self.arms.append(np.dot(Qs[n], factors[n]))
        X_hat = tl.tucker_to_tensor(self.core_tensor, self.arms)
        error = self.X-X_hat
        error = np.linalg.norm(error.reshape(np.size(error),1), 'fro')
        mse = error/np.size(self.X)
        return self.arms, self.core_tensor, error, mse

if __name__ == "__main__":
    pass











