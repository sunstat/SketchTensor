import numpy as np
import pandas as pd
from simulation import Simulation
from util import TensorInfoBucket
from util import RandomInfoBucket
import tensorly as tl

'''
begin global variables
'''
n=100
ratios = [1.2, 1.5, 2.0, 2.5]
ranks = [1]
dim = 4
ks = np.arange(2, 12,2)
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

mse = { 'ho_svd': np.zeros([len(gen_types), len(ranks), len(ratios), len(ks)]),
        'simu_two_pass': np.zeros([len(gen_types), len(ranks), len(ratios), len(ks)]),
        'simu_one_pass': np.zeros([len(gen_types), len(ranks), len(ratios), len(ks)])
       }

values = np.zeros([len(ranks), len(ratios), len(ks), len(gen_types)])
it = np.nditer(values, flags=['multi_index'])

'''
gen_type, k, ratio, rank = 'id', 2, 1.5, 1
simu = Simulation(TensorInfoBucket([n,n,n], k = k, rank = rank, s=int(k*ratio)),\
        RandomInfoBucket(random_seed = 1), gen_typ = gen_type, noise_level=0)

simu.run('one_pass', simu_runs)
'''
tl.set_backend('numpy')


print(len(gen_types), len(ranks), len(ratios), len(ks))
for i1,gen_type in enumerate((gen_types)): 
    for i2,rank in enumerate((ranks)):
        for i3,ratio in enumerate((ratios)): 
            for i4,k in enumerate((ks)): 
                    print(i1,gen_type,';', i2,rank,';',i3, ratio,';',i4, k)
                    simu = Simulation(np.repeat(n,dim), rank, k, int(k*ratio),\
                        RandomInfoBucket(random_seed = 1),gen_type,0)
                    _, mse1 = simu.one_pass()
                    temp = mse['simu_one_pass']
                    print('mse = ',mse1)
                    temp[i1,i2,i3,i4] = mse1 
                    mse['simu_one_pass'] = temp


print(mse['simu_one_pass'])

tensor_shape, rank, k, s, Rinfo_bucket, gen_typ, noise_level



'''


while not it.finished:
    rank, ratio, k, gen_type = get_info(it.multi_index)
    print("=============")
    
    print("======")
    running_times, mse_arr, _ = simu.run('one_sketch', simu_runs)
    print(mse_arr)
    exit(0)
if __name__ == "__main___":

'''



