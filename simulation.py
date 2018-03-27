import numpy as np
from scipy import fftpack
import tensorly as tl
from util import square_tensor_gen, TensorInfoBucket, RandomInfoBucket, eval_mse, eval_rerr
from sketch import Sketch
import time
from tensorly.decomposition import tucker
from sketch_recover import SketchTwoPassRecover
from sketch_recover import SketchOnePassRecover

class Simulation(object):
    '''
    In this simulation, we only experiment with the square design and Gaussian 
    randomized linear map. We use the same random_seed for generating the data matrix and the arm matrix
    '''
    def __init__(self, tensor_shape, rank, k, s, Rinfo_bucket, gen_typ, noise_level):
        tl.set_backend('numpy')
        self.tensor_shape, self.rank, self.k, self.s = tensor_shape, rank, k, s
        self.n = self.tensor_shape[0]
        self.dim = len(self.tensor_shape)
        self.std, self.typ, self.random_seed, self.sparse_factor =  Rinfo_bucket.get_info()
        self.total_num = np.prod(self.tensor_shape)
        self.gen_typ = gen_typ
        self.noise_level = noise_level
        self.Rinfo_bucket = Rinfo_bucket

    def ho_svd(self):
        X = square_tensor_gen(self.n, self.rank, dim=self.dim, typ=self.gen_typ,\
         noise_level=self.noise_level, seed = self.random_seed)
        start_time = time.time()
        core, tucker_factors = tucker(X, ranks=[self.rank for _ in range(self.dim)], init='random')
        X_hat = tl.tucker_to_tensor(core, tucker_factors)
        running_time = time.time() - start_time
        rerr = eval_rerr(X,X_hat)
        return (-1, running_time), rerr

    def two_pass(self):
        X = square_tensor_gen(self.n, self.rank, dim=self.dim, typ=self.gen_typ, \
            noise_level=self.noise_level, seed = self.random_seed)
        start_time = time.time()
        sketch = Sketch(X, self.k, random_seed = self.random_seed)
        sketchs, _, = sketch.get_sketchs()
        sketch_time = time.time() - start_time
        start_time = time.time()
        sketch_two_pass = SketchTwoPassRecover(X, sketchs, np.repeat(self.rank,self.dim))
        X_hat,_,_ =  sketch_two_pass.recover()
        recover_time = time.time() - start_time
        rerr = eval_rerr(X,X_hat)
        return (sketch_time, recover_time), rerr
    def one_pass(self, store_phis = True):
        X = square_tensor_gen(self.n, self.rank, dim=self.dim, typ=self.gen_typ, \
            noise_level=self.noise_level, seed = self.random_seed)
        start_time = time.time()
        sketch = Sketch(X, self.k, random_seed = self.random_seed, s = self.s, store_phis = store_phis)
        sketchs, core_sketch = sketch.get_sketchs() 
        sketch_time = time.time() - start_time
        start_time = time.time()
        sketch_one_pass = SketchOnePassRecover(sketchs,core_sketch,\
            TensorInfoBucket(self.tensor_shape,self.k,np.repeat\
                (self.rank,self.dim), self.s),self.Rinfo_bucket,\
            sketch.get_phis())
        X_hat, _, _  = sketch_one_pass.recover()

        recover_time = time.time() - start_time
        rerr = eval_rerr(X,X_hat)
        return (sketch_time, recover_time), rerr

import matplotlib 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    n = 200
    k = 12  
    rank = 5 
    dim = 3 
    s = 80 
    tensor_shape = np.repeat(n,dim)
    noise_level = 0.01
    gen_typ = 'id' 
    Rinfo_bucket = RandomInfoBucket(random_seed = 1)
    '''
    simu = Simulation(tensor_shape, rank, k, s, Rinfo_bucket, gen_typ, noise_level)
    _, rerr = simu.ho_svd()
    print('ho_svd rerr:', rerr)
    _, rerr = simu.two_pass() 
    print('two_pass:', rerr) 
    _, rerr = simu.one_pass()
    print('one_pass:', rerr)
    ''' 

    noise_levels = (np.float(10)**(np.arange(-10,2,2))) 
    ho_svd_rerr = np.zeros(len(noise_levels))
    two_pass_rerr = np.zeros(len(noise_levels))
    one_pass_rerr = np.zeros(len(noise_levels))
    one_pass_rerr_ns = np.zeros(len(noise_levels))

    for idx, noise_level in enumerate(noise_levels): 
        print('Noise_level:', noise_level)
        simu = Simulation(tensor_shape, rank, k, s, Rinfo_bucket, gen_typ, noise_level)
        _, rerr = simu.ho_svd()
        #print('ho_svd rerr:', rerr) 
        ho_svd_rerr[idx] = rerr 

        _, rerr = simu.two_pass() 
        #print('two_pass:', rerr) 
        two_pass_rerr[idx] = rerr

        _, rerr = simu.one_pass()
        #print('one_pass:', rerr)
        one_pass_rerr[idx] = rerr

        _, rerr = simu.one_pass(store_phis = False) 
        one_pass_rerr_ns[idx]  = rerr


    print("identity design with varying noise_level")
    print("noise_levels", noise_levels)
    print("ho_svd", ho_svd_rerr)
    print("two_pass", two_pass_rerr)
    print("one_pass", one_pass_rerr)
    print("one_pass_ns", one_pass_rerr_ns)

    plt.subplot(3,1,1)
    plt.plot(noise_levels,ho_svd_rerr,label = 'ho_svd')
    plt.title('ho_svd')
    plt.subplot(3,1,2)
    plt.plot(noise_levels,two_pass_rerr, label = 'two_pass')
    plt.title('two_pass')
    plt.subplot(3,1,3) 
    plt.plot(noise_levels,one_pass_rerr, label = 'one_pass') 
    plt.title('one_pass')
    plt.show()



