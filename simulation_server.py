import numpy as np
from scipy import fftpack
import tensorly as tl
from util import square_tensor_gen, TensorInfoBucket, RandomInfoBucket, eval_rerr
from sketch import Sketch
import time
from tensorly.decomposition import tucker
from sketch_recover import SketchTwoPassRecover
from sketch_recover import SketchOnePassRecover
import matplotlib 
import matplotlib.pyplot as plt
from simulation import Simulation
import matplotlib.ticker as ticker
import pickle 
plt.style.use('seaborn-paper')


def sim_name(gen_type,r,noise_level,dim): 
    '''
    Create a file name for a simulation setting
    '''
    if noise_level == 0: 
        noise = "no"
    else: 
        noise = str(int(np.log10(noise_level)))
    return "data/typ"+gen_type+"_r"+str(r)+"_noise"+noise+"_dim"+str(dim)

def run_nssim(gen_type,r,noise_level, ns = np.arange(100,101,100), dim = 3, \
    sim_runs = 1,random_seed = 1): 
    '''
    Given a data generation procedure, perform numerical simulations and return 
    the relative error. 
    '''
    sim_list = []
    for id, n in enumerate(ns): 
        if gen_type in ['id','lk']: 
            ks =np.arange(r, int(n/2),int(n/20)) 
        else:
            ks = np.arange(r,int(n/10),int(n/100))
        hooi_rerr = np.zeros((sim_runs, len(ks)))
        two_pass_rerr = np.zeros((sim_runs,len(ks)))
        one_pass_rerr = np.zeros((sim_runs,len(ks)))
        for i in range(sim_runs): 
            for idx, k in enumerate(ks): 
                simu = Simulation(np.repeat(n,dim), r, k, 2*k+1, \
                    RandomInfoBucket(random_seed), gen_type, noise_level)
                _, rerr = simu.hooi()
                hooi_rerr[i,idx] = rerr 
                _, rerr = simu.two_pass() 
                two_pass_rerr[i,idx] = rerr
                _, rerr = simu.one_pass()
                one_pass_rerr[i,idx] = rerr 
        sim_list.append([two_pass_rerr,one_pass_rerr,hooi_rerr])
    pickle.dump( sim_list, open(sim_name(gen_type,r,noise_level,dim) \
        +".pickle", "wb" ) )
    return sim_list

if __name__ == '__main__':
    run_nssim('sed',5,0.01,np.arange(200,601,200)) 
    run_nssim('fed',5,0.01,np.arange(200,601,200)) 




