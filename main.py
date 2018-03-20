import numpy as np
import pandas as pd
from simulation import Simulation
from util import TensorInfoBucket
from util import RandomInfoBucket


'''
begin global variables
'''
n=1000
ratios = [1.2, 1.5, 2.0, 2.5]
ranks = [1,5]
ks = np.arange(5, 100, 2)
gen_types = ['id', 'id1', 'id2', 'spd', 'fpd', 'sed', 'fed', 'lk']
mse = { 'ho_svd': np.zeros([len(ranks), len(ratios), len(ks), len(gen_types)]),
        'simu_two_pass': np.zeros([len(ranks), len(ratios), len(ks), len(gen_types)]),
        'simu_one_pass': np.zeros([len(ranks), len(ratios), len(ks), len(gen_types)])
       }

'''
end global variables
'''

def get_info(index):
    rank = ranks[index[0]]
    ratio = ratios[index[1]]
    k = ks[index[3]]
    gen_type = gen_types[index[4]]
    return rank, ratio, k, gen_type

it = np.nditer(mse, flags=['multi_index'])


while not it.finished:
    rank, ratio, k, gen_type = get_info(it.multi_index)
    simu = Simulation(TensorInfoBucket([n,n,n], k = k, rank = rank, s=int(k*ratio)),\
        RandomInfoBucket(random_seed = 1), gen_typ = gen_type, noise_level=0)
    simu.run('ho_svd', )










