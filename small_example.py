import numpy as np
import pandas as pd
from simulation import Simulation
from util import TensorInfoBucket
from util import RandomInfoBucket


'''
begin global variables
'''
n=100
ratios = [1.2, 1.5, 2.0, 2.5]
ranks = [1, 5]
ks = np.arange(5, 12, 2)
gen_types = ['id', 'id1', 'id2']
mse = { 'ho_svd': np.zeros([len(ranks), len(ratios), len(ks), len(gen_types)]),
        'simu_two_pass': np.zeros([len(ranks), len(ratios), len(ks), len(gen_types)]),
        'simu_one_pass': np.zeros([len(ranks), len(ratios), len(ks), len(gen_types)])
       }
simu_runs = 1

'''
end global variables
'''

def get_info(index):
    rank = ranks[index[0]]
    ratio = ratios[index[1]]
    k = ks[index[2]]
    gen_type = gen_types[index[3]]
    return rank, ratio, k, gen_type


values = np.zeros([len(ranks), len(ratios), len(ks), len(gen_types)])
it = np.nditer(values, flags=['multi_index'])


while not it.finished:
    rank, ratio, k, gen_type = get_info(it.multi_index)
    print("=============")
    simu = Simulation(TensorInfoBucket([n,n,n], k = k, rank = rank, s=int(k*ratio)),\
        RandomInfoBucket(random_seed = 1), gen_typ = gen_type, noise_level=0)
    print("======")
    running_times, mse_arr, _ = simu.run('one_sketch', simu_runs)
    print(mse_arr)
    exit(0)



if __name__ == "__main___":





