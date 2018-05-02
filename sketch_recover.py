import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from util import RandomInfoBucket
from util import random_matrix_generator
from util import generate_super_diagonal_tensor  

class SketchTwoPassRecover(object):
    def __init__(self, X, sketchs, ranks):
        tl.set_backend('numpy')
        self.arms = []
        self.core_tensor = None
        self.X = X
        self.sketchs = sketchs
        self.ranks = ranks

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
        core_tensor, factors = tucker(core_tensor, ranks=self.ranks)
        self.core_tensor = core_tensor

        #arm[n] = Q.T*factors[n]
        for n in range(len(factors)):
            self.arms.append(np.dot(Qs[n], factors[n]))
        X_hat = tl.tucker_to_tensor(self.core_tensor, self.arms)
        return X_hat, self.arms, self.core_tensor 

class SketchOnePassRecover(object):

    @staticmethod
    def get_phis(Rinfo_bucket, tensor_shape, ks, ss):
        total_num = np.prod(tensor_shape)
        phis = []
        # generate omegas which we do not need store
        for i in range(len(tensor_shape)):
            n1 = tensor_shape[i]
            n1 = total_num//n1
            _ = random_matrix_generator(n1, ks[i], Rinfo_bucket)

        for i in range(len(tensor_shape)):
<<<<<<< HEAD
            phis.append(random_matrix_generator(s, tensor_shape[i], Rinfo_bucket))
=======
            phis.append(random_matrix_generator(ss[i], tensor_shape[i], \
                Rinfo_bucket))
>>>>>>> 6821aadcf5efcc9a604a932992d1849269e5c434
        return phis

    def __init__(self, sketchs, core_sketch, Tinfo_bucket, Rinfo_bucket,phis = []):
        tl.set_backend('numpy')
        self.arms = []
        self.core_tensor = None
        self.sketchs = sketchs
        # Note get_info extract some extraneous information
<<<<<<< HEAD
        self.tensor_shape, self.k, self.ranks,_ = Tinfo_bucket.get_info()
        self.s = core_sketch.shape[1]
=======
        self.tensor_shape, self.ks, self.ranks,self.ss= Tinfo_bucket.get_info()
>>>>>>> 6821aadcf5efcc9a604a932992d1849269e5c434
        self.Rinfo_bucket = Rinfo_bucket
        self.phis = phis
        self.core_sketch = core_sketch

    def recover(self):
        if self.phis == []:
<<<<<<< HEAD
            phis = SketchOnePassRecover.get_phis(self.Rinfo_bucket, tensor_shape = self.tensor_shape, k = self.k, s = self.s)
=======
            phis = SketchOnePassRecover.get_phis(self.Rinfo_bucket, \
                tensor_shape = self.tensor_shape, ks = self.ks, ss = self.ss)
>>>>>>> 6821aadcf5efcc9a604a932992d1849269e5c434
        else: 
            phis = self.phis 
        Qs = []
        for sketch in self.sketchs:
            Q, _ = np.linalg.qr(sketch)
            Qs.append(Q)
        self.core_tensor = self.core_sketch
        dim = len(self.tensor_shape)
        for mode_n in range(dim):
<<<<<<< HEAD
            self.core_tensor = tl.tenalg.mode_dot(self.core_tensor, np.linalg.pinv(np.dot(phis[mode_n], Qs[mode_n])), mode=mode_n)
=======
            self.core_tensor = tl.tenalg.mode_dot(self.core_tensor, \
                np.linalg.pinv(np.dot(phis[mode_n], Qs[mode_n])), mode=mode_n)
>>>>>>> 6821aadcf5efcc9a604a932992d1849269e5c434
        core_tensor, factors = tucker(self.core_tensor, ranks= self.ranks)
        self.core_tensor = core_tensor
        for n in range(dim):
            self.arms.append(np.dot(Qs[n], factors[n]))
        X_hat = tl.tucker_to_tensor(self.core_tensor, self.arms)
        return X_hat, self.arms, self.core_tensor



from util import *
from sketch import *

if __name__ == "__main__":
    tl.set_backend('numpy')
    
<<<<<<< HEAD
    s = 80
    k = 12 
    n = 200 
=======
    ss = [80,90,100]
    ks = [12,13,14]
    n = 200  
>>>>>>> 6821aadcf5efcc9a604a932992d1849269e5c434
    dim = 3 
    rank = 5 
    ranks = np.repeat(rank,dim) 
    size = np.repeat(n,dim) 
<<<<<<< HEAD
    X = square_tensor_gen(n, rank, dim=dim, typ='id', noise_level = 0.001)

    print(tl.unfold(X, mode=1).shape)
    tensor_sketch = Sketch(X, k, random_seed = 1, s = s, typ = 'u', sparse_factor = 0.1,store_phis = True)
=======
    X, X0= square_tensor_gen(n, rank, dim=dim, typ='id', noise_level = 0.001)

    tensor_sketch = Sketch(X, ks, random_seed = 1, ss = ss, typ = 'u', \
        sparse_factor = 0.1,store_phis = True)
>>>>>>> 6821aadcf5efcc9a604a932992d1849269e5c434
    phis = tensor_sketch.get_phis()
    sketchs, core_sketch = tensor_sketch.get_sketchs() 
    two_pass = SketchTwoPassRecover(X,sketchs,ranks)
    X_hat,_ ,_ = two_pass.recover()
<<<<<<< HEAD
    print('two_pass:',eval_rerr(X,X_hat))

    one_pass0 = SketchOnePassRecover(sketchs,core_sketch,TensorInfoBucket(size,k,ranks,s),RandomInfoBucket(random_seed = 1))
    X_hat,_ ,_ = one_pass0.recover()
    print('one_pass:',eval_rerr(X,X_hat))
    one_pass = SketchOnePassRecover(sketchs,core_sketch,TensorInfoBucket(size,k,ranks,s),RandomInfoBucket(random_seed = 1),phis = phis)
    X_hat,_ ,_ = one_pass.recover()
    print('one_pass_no_rs:',eval_rerr(X,X_hat))
=======
    print('two_pass:',eval_rerr(X,X_hat,X0))

    one_pass0 = SketchOnePassRecover(sketchs,core_sketch,\
        TensorInfoBucket(size,ks,ranks,ss),RandomInfoBucket(random_seed = 1))
    X_hat,_ ,_ = one_pass0.recover()
    print('one_pass:',eval_rerr(X,X_hat,X0))
    one_pass = SketchOnePassRecover(sketchs,core_sketch,TensorInfoBucket\
        (size,ks,ranks,ss),RandomInfoBucket(random_seed = 1),phis = phis)
    X_hat,_ ,_ = one_pass.recover()
    print('one_pass_no_rs:',eval_rerr(X,X_hat,X0))
>>>>>>> 6821aadcf5efcc9a604a932992d1849269e5c434


