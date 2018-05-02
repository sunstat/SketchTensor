import numpy as np
from scipy import fftpack
import tensorly as tl
from util import square_tensor_gen, TensorInfoBucket, RandomInfoBucket, eval_rerr
from sketch import Sketch
import time
from tensorly.decomposition import tucker
from sketch_recover import SketchTwoPassRecover
from sketch_recover import SketchOnePassRecover 

def hooi_data(X, ranks, random_seed = 1):  
    start_time = time.time()
    core, tucker_factors = tucker(X, ranks, init = 'svd')  
    X_hat = tl.tucker_to_tensor(core, tucker_factors) 
    running_time = time.time() - start_time 
    rerr = eval_rerr(X,X_hat,X) 
    return (-1, running_time), rerr

def two_pass_data(X, ranks, ks, random_seed = 1):  
    start_time = time.time()
    sketch = Sketch(X, ks, random_seed = random_seed) 
    sketchs, _ = sketch.get_sketchs() 
    sketch_time = time.time() - start_time 
    start_time = time.time()
    sketch_two_pass = SketchTwoPassRecover(X, sketchs, ranks) 
    X_hat, _, _ = sketch_two_pass.recover() 
    recover_time = time.time() - start_time 
    rerr = eval_rerr(X,X_hat,X) 
    return (sketch_time, recover_time), rerr 

def one_pass_data(X, ranks, ks, ss, random_seed = 1, store_phis = True): 
    start_time = time.time() 
    sketch = Sketch(X, ks, random_seed = 1,ss =ss, store_phis = store_phis) 
    sketchs, core_sketch = sketch.get_sketchs()  
    sketch_time = time.time() - start_time 
    start_time = time.time()
    sketch_one_pass = SketchOnePassRecover(sketchs, core_sketch, \
        TensorInfoBucket(X.shape, ks,ranks,ss),RandomInfoBucket(random_seed \
            = random_seed), sketch.get_phis()) 
    X_hat, _, _ = sketch_one_pass.recover() 
    recover_time = time.time() - start_time 
    rerr = eval_rerr(X, X_hat, X) 
    return (sketch_time, recover_time), rerr 



if __name__ == '__main__':

    # Test it for square data
    n = 100
    k = 10  
    rank = 5 
    dim = 3 
    s = 2*k+1
    ranks = np.repeat(rank,dim)
    ks = np.repeat(k,dim)
    ss = np.repeat(s,dim)
    tensor_shape = np.repeat(n,dim)
    noise_level = 0.01
    gen_typ = 'lk' 
    X, X0 = square_tensor_gen(n, rank, dim, gen_typ, \
            noise_level, seed = 1) 
    _, rerr = hooi_data(X, ranks)
    print(rerr) 

    _, rerr = two_pass_data(X, ranks, ks)
    print(rerr) 

    _, rerr = one_pass_data(X, ranks, ks,ss)
    print(rerr) 


    # Test it for data with unequal side length

    ranks = np.array((5, 10, 15))
    dim = 3 
    ns = np.array((100,200,300)) 
    ks = np.array((15, 20, 25))
    ss = 2*ks + 1 
    core_tensor = np.random.uniform(0,1,ranks)
    arms = []
    tensor = core_tensor
    for i in np.arange(dim):
        arm = np.random.normal(0,1,size = (ns[i],ranks[i]))
        arm, _ = np.linalg.qr(arm)
        arms.append(arm)
        tensor = tl.tenalg.mode_dot(tensor, arm, mode=i)
    true_signal_mag = np.linalg.norm(core_tensor)**2
    noise = np.random.normal(0, 1, ns)
    X = tensor + noise*np.sqrt((noise_level**2)*true_signal_mag/np.product\
        (np.prod(ns)))

    _, rerr = hooi_data(X, ranks)
    print(rerr) 

    _, rerr = two_pass_data(X, ranks, ks)
    print(rerr) 

    _, rerr = one_pass_data(X, ranks, ks,ss)
    print(rerr) 













