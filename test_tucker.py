import numpy as np
from scipy import fftpack
import tensorly as tl
from util import square_tensor_gen, TensorInfoBucket, RandomInfoBucket, eval_mse, eval_rerr
from sketch import Sketch
import time
from tensorly.decomposition import tucker
from sketch_recover import SketchTwoPassRecover
from sketch_recover import SketchOnePassRecover
from sklearn.utils.extmath import randomized_svd


def run_hosvd(X,ranks): 
    arms = [] 
    core = X
    for mode in range(X.ndim):
        U,_,_ = randomized_svd(tl.unfold(X,mode),ranks[mode])
        arms.append(U) 
        core = tl.tenalg.mode_dot(core, U.T,mode) 
    return core, arms

if __name__ == '__main__':
    n = 100
    k = 10  
    rank = 5 
    dim = 3 
    s = 20 
    tensor_shape = np.repeat(n,dim)
    noise_level = 0.01
    gen_typ = 'lk' 
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
    X, X0 = square_tensor_gen(n, rank, dim=dim, typ=gen_typ,\
         noise_level= noise_level, seed = 1)
    #core, tucker_factors = tucker(X, ranks=[rank for _ in range(dim)], init='svd')
    
    core, tucker_factors = run_hosvd(X,ranks=[90 for _ in range(dim)])
    Xhat = tl.tucker_to_tensor(core, tucker_factors)

    print(X)
    print(eval_rerr(X,Xhat,X0))
    print(np.linalg.norm((X-X0).reshape(X.size,1),'fro')/np.linalg.norm(X0.reshape(X.size,1), 'fro'))


