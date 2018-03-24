import tensorly as tl
import numpy as np
from operator import mul
from util import random_matrix_generator
from util import RandomInfoBucket
from util import square_tensor_gen


class Sketch(object):

    @staticmethod
    def sketch_arm_rm_generator(tensor_shape, reduced_dim, random_seed, typ='g', sparse_factor=0.1):
        '''
        :param tensor_shape: shape of the tensor, an 1-d array
        :param reduced_dim: k, the dimension that the core tensor will be reduced into 
        '''
        total_num = np.prod(tensor_shape)
        for n in range(len(tensor_shape)):
            n1 = total_num//tensor_shape[n] # I_(-n)
            yield random_matrix_generator(n1, reduced_dim, RandomInfoBucket(std=1, typ=typ, random_seed = random_seed,
                                                                            sparse_factor = sparse_factor))

    @staticmethod
    def sketch_core_rm_generator(tensor_shape, reduced_dim, random_seed, typ='g', sparse_factor=0.1):
        
        for n in range(len(tensor_shape)):
            yield random_matrix_generator(tensor_shape[n], reduced_dim,\
                        RandomInfoBucket(std=1, typ=typ, random_seed= random_seed, sparse_factor=sparse_factor))


    @staticmethod 
    def sketch_arm_rm(tensor_shape, reduced_dim, random_seed, typ = 'g', sparse_factor = 0.1) :
        arm_rm = []
        total_num = np.prod(tensor_shape)
        for n in range(len(tensor_shape)):
            n1 = total_num//tensor_shape[n] # I_(-n)
            arm_rm.append(random_matrix_generator(n1, reduced_dim, RandomInfoBucket(std=1, typ=typ, random_seed = random_seed,
                                                                            sparse_factor = sparse_factor)))
        return arm_rm

    @staticmethod 
    def sketch_core_rm(tensor_shape, reduced_dim, random_seed, typ = 'g', sparse_factor = 0.1):
        core_rm = []

        for n in range(len(tensor_shape)): 
            core_rm.append(random_matrix_generator(tensor_shape[n], reduced_dim,\
                        RandomInfoBucket(std=1, typ=typ, random_seed= random_seed, sparse_factor=sparse_factor)))
        return core_rm


    def __init__(self, X, k, random_seed, s = -1, typ = 'g', sparse_factor = 0.1,store_rm = False) :
        '''
        :param X: tensor being skeched
        :param k:
        :param s: s>k
        :param random_seed: random_seed
        :param sparse_factor: only typ == 'sp', p matters representing the sparse factor
        '''
        tl.set_backend('numpy')
        self.X = X
        self.N = len(X.shape)
        self.s = s
        self.k = k
        self.typ = typ
        self.sparse_factor = sparse_factor
        self.sketchs = []
        self.random_seed = random_seed
        self.core_sketch = X
        self.tensor_shape = X.shape
        self.arm_rm = []
        self.core_rm = []

        # set the random seed for following procedure
        np.random.seed(random_seed)

        if not store_rm:
            
            rm_generator = Sketch.sketch_arm_rm_generator(self.tensor_shape,  reduced_dim=self.k, random_seed = random_seed,
                                                              typ=self.typ, sparse_factor=self.sparse_factor)
            mode_n = 0
            for rm in rm_generator:
                self.sketchs.append(np.dot(tl.unfold(self.X, mode=mode_n), rm))
                mode_n += 1

            if self.s != -1:
                rm_generator = Sketch.sketch_core_rm_generator(self.tensor_shape,  reduced_dim=self.s, random_seed = random_seed,
                                                                  typ=self.typ, sparse_factor=self.sparse_factor)
                mode_n = 0
                for rm in rm_generator:
                    self.core_sketch = tl.tenalg.mode_dot(self.core_sketch, rm.T, mode=mode_n)
                    mode_n += 1
        else: 
            self.arm_rm = Sketch.sketch_arm_rm(self.tensor_shape,  reduced_dim=self.k, random_seed = random_seed,
                                                              typ=self.typ, sparse_factor=self.sparse_factor)
            for mode_n, rm in enumerate(self.arm_rm):
                self.sketchs.append(np.dot(tl.unfold(self.X, mode=mode_n), rm))    
            if self.s != -1: 
                self.core_rm = Sketch.sketch_core_rm(self.tensor_shape,  reduced_dim=self.s, random_seed = random_seed,
                                                                  typ=self.typ, sparse_factor=self.sparse_factor)
                for mode_n, rm in enumerate(self.core_rm): 
                    self.core_sketch = tl.tenalg.mode_dot(self.core_sketch, rm.T, mode=mode_n)
                    

    def get_sketchs(self):
        return self.sketchs, self.core_sketch
        [self for x in arm_rm]
    def get_rm(self):
        return self.arm_rm, self.core_rm

    def get_Rinfo_bucket(self):
        return RandomInfoBucket(1,self.typ, self.random_seed, self.sparse_factor)

if __name__ == "__main__":
    tl.set_backend('numpy')
    X = square_tensor_gen(10, 3, dim=3, typ='spd', noise_level=0.1)
    print(tl.unfold(X, mode=1).shape)
    tensor_sketch = Sketch(X = X, k = 5, random_seed = 1, s = -1, typ = 'g', sparse_factor = 0.1, store_rm = True)
    sketchs, core_sketch  = tensor_sketch.get_sketchs()
    print(len(sketchs))
    for sketch in sketchs:
        print(sketch)
    print(core_sketch.shape)

    #=======================
    tensor_sketch = Sketch(X, 5, random_seed=1, s=6, typ='g', sparse_factor=0.1,store_rm = True)
    sketchs, core_sketch = tensor_sketch.get_sketchs()
    print(len(sketchs))
    for sketch in sketchs:
        print(sketch)
    print(core_sketch.shape)

    print("random map")
    arm_rm, core_rm = tensor_sketch.get_rm()
    print(core_rm)
