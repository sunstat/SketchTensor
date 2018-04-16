import numpy as np
from scipy import fftpack
import tensorly as tl
from util import square_tensor_gen, TensorInfoBucket, RandomInfoBucket, eval_mse, eval_rerr
from sketch import Sketch
import time
from tensorly.decomposition import tucker
from sketch_recover import SketchTwoPassRecover
from sketch_recover import SketchOnePassRecover
import matplotlib 
import matplotlib.pyplot as plt
from simulation import Simulation
import pickle 
plt.style.use('seaborn-paper')
get_ipython().magic('matplotlib inline')

def sim_name(gen_type,r,noise_level,dim): 
    return "data/typ"+gen_type+"_r"+str(r)+"_noise"+str(int(np.log10(0.01)))+"_dim"+str(dim)


def run_nssim(gen_type,r,noise_level, ns = np.arange(100,101,100), dim = 3, sim_runs = 1,random_seed = 1): 
    sim_list = []
    for id, n in enumerate(ns): 
        ks =np.arange(r, int(n/2),int(n/20)) 
        ho_svd_rerr = np.zeros((sim_runs, len(ks)))
        two_pass_rerr = np.zeros((sim_runs,len(ks)))
        one_pass_rerr = np.zeros((sim_runs,len(ks)))
        for i in range(sim_runs): 
            for idx, k in enumerate(ks): 
                simu = Simulation(np.repeat(n,dim), r, k, 2*k+1, RandomInfoBucket(random_seed), gen_type, noise_level)
                _, rerr = simu.ho_svd()
                ho_svd_rerr[i,idx] = rerr 
                _, rerr = simu.two_pass() 
                two_pass_rerr[i,idx] = rerr
                _, rerr = simu.one_pass()
                one_pass_rerr[i,idx] = rerr 
        sim_list.append([two_pass_rerr,one_pass_rerr,ho_svd_rerr])
    pickle.dump( sim_list, open(sim_name(gen_type,r,noise_level,dim) +".pickle", "wb" ) )
    return sim_list

def plot_nssim(gen_type,r,noise_level,name, ns = np.arange(100,101,100), dim = 3, sim_runs = 1,random_seed = 1): 
    sim_list = pickle.load( open( sim_name(gen_type,r,noise_level,dim)+".pickle", "rb" ) ) 
    plt.close()
    plt.figure(figsize = (len(ns)*5,5))
    for plot_id, n in enumerate(ns): 
        ks =np.arange(r, int(n/2),int(n/20)) 
        plt.subplot(1,len(ns),plot_id+1) 
        plt.plot(ks/n,np.mean(sim_list[plot_id][0],0), label = 'two_pass',linestyle = '--',marker = 'X')
        plt.title('two_pass')
        plt.plot(ks/n,np.mean(sim_list[plot_id][1],0), label = 'one_pass', marker = 's',markeredgewidth=1,markeredgecolor='orange', markerfacecolor='None') 
        plt.title('one_pass') 
        plt.plot(ks/n,np.mean(sim_list[plot_id][2],0), label = 'hosvd', linestyle = ':', marker = 'o', markeredgewidth=1,markeredgecolor='g', markerfacecolor='None') 
        plt.title('hosvd') 
        plt.legend(loc = 'best')
        plt.title("I = %s"%(n))
        plt.xlabel('k/I')
        plt.ylabel('log(Relative Error)')
        plt.yscale('log')
        plt.axis('equal')
        xmin, xmax = plt.xlim()  
        plt.xlim((0, xmax))  
        ymin, ymax = plt.ylim()  
        plt.ylim((ymin, ymax))   
    plt.tight_layout()
    plt.savefig('plots/'+name)
    plt.show()

if __name__ == '__main__':
    run_nssim('id',5,0.01,np.arange(200,601,200)) 
    run_nssim('id',5,0.1,np.arange(200,601,200)) 
    run_nssim('id',5,1,np.arange(200,601,200)) 
    run_nssim('id',1,0.01,np.arange(200,601,200)) 
    run_nssim('spd',5,0.01,np.arange(200,601,200)) 
    run_nssim('fpd',5,0.01,np.arange(200,601,200)) 
    run_nssim('sed',5,0.01,np.arange(200,601,200)) 
    run_nssim('fed',5,0.01,np.arange(200,601,200)) 
    run_nssim('lk',5,0.01,np.arange(200,601,200)) 


