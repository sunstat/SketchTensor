from simulation import *
from sketch import *
from sketch_recover import * 
from util import *


n = 200
k = 12  
rank = 5 
dim = 3 
s = 80 
tensor_shape = np.repeat(n,dim)
noise_level = 0.01
gen_typ = 'id' 
Rinfo_bucket = RandomInfoBucket(random_seed = 1)

simu = Simulation(tensor_shape, rank, k, s, Rinfo_bucket, gen_typ, noise_level)
_, rerr = simu.ho_svd()
print('ho_svd rerr:', rerr)
_, rerr = simu.two_pass() 
print('two_pass:', rerr) 
_, rerr = simu.one_pass()
print('one_pass:', rerr)
_, rerr = simu.one_pass(store_phis = False)
print('one_pass_xphis:', rerr)




'''
    noise_levels = (np.float(10)**(np.arange(-10,2,2))) 
    ho_svd_rerr = np.zeros(len(noise_levels))
    two_pass_rerr = np.zeros(len(noise_levels))
    one_pass_rerr = np.zeros(len(noise_levels))

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
'''

