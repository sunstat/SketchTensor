import numpy as np
from scipy import fftpack
import tensorly as tl
from util import *

def tensorGen(size,r,typ = 'id',snr = 0.1):  
    types = set(['id', 'lowrank'])
    assert typ in types, "please set your type of tensor correctly" 
    if typ == 'id': 
        identity = np.zeros(size) 
        for i in np.arange(min(r)): 
            identity[(np.repeat(i,len(size)))] = 1
        noise = np.random.normal(size = size) 
        return identity+noise*np.sqrt(snr*min(r)/np.product(size)) 
    if typ == "lowrank": 
        core = np.random.normal(size = r) 
        for i in np.arange(len(size)):
            arm = np.random.normal(size = (size[i],r[i]))
            print('arm' +size(arm))
            print('core'+size(core))
            core = tl.tenalg.mode_dot(core,arm,mode = i) 
        return core  

if __name__ == '__main__':
    print(tensorGen((5,6,7),(2,3,4),typ = 'id') )
    print(tensorGen((5,6,7),(2,3,4),typ = 'lowrank') )