import numpy as np
from scipy import fftpack

def randomMatrixGenerator(m, n, std = 1, typ ='g',rand_seed = None, sparse_factor = 0.1):
    '''
    :param m: # of rows in the matrix
    :param n: # of columns in the matrix
    :param std: standard deviation
    :param typ: 'g' for gauss, 'u' for uniform, 's' for Scrambled SRFT; 'sp' 
    for Sparse Sign
    :return: the generated matrix
    '''
    if rand_seed:
        np.random.seed(rand_seed)

    if (typ == 'g'): 
        return np.random.normal(size = (m,n))*std
    elif (typ == 'u'):
        return np.random.uniform(low = -1, high = 1, size = (m,n))/3*std 
    elif (typ == 'sp'): 
        return np.random.binomial(n = 1,p = sparse_factor,size = (m,n))*np.random.uniform(low = -1, high = 1, size = (m,n))/3*std 
    elif (typ == 's'): 
        # Scramble SSRFT map: R*F*PI*F*PI', where R: random uniform with std, 
        # F: Discrete cousine transform; PI: Signed permutation matrix
        i = np.identity(n)
        rr = np.range(n)
        np.random.shuffle(rr)
        perm = np.take(i, rr, axis=0) 
        return (np.random.uniform(low = -1, high = 1, size = (m,n))/3*std).dot(fftpack.dct(perm)).dot(fftpack.dct(perm.T))
    else: 
        print('Please enter a valid type for the random matrix: g (Gaussian)/u (uniform)/sp (sparse sign)/s (SSRFT)!')
