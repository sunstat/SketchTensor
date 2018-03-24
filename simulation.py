import numpy as np
from scipy import fftpack
import tensorly as tl
from util import square_tensor_gen
from util import TensorInfoBucket
from util import RandomInfoBucket
from util import eval_mse
from sketch import Sketch
import time
from tensorly.decomposition import tucker
from sketch_recover import SketchTwoPassRecover
from sketch_recover import SketchOnePassRecover

class Simulation(object):
    def __init__(self, Tinfo_bucket, Rinfo_bucket, gen_typ, noise_level):
        tl.set_backend('numpy')
        self.tensor_shape, self.k, self.rank, self.s = Tinfo_bucket.get_info()
        self.n = self.tensor_shape[0]
        self.dim = len(self.tensor_shape)
        self.std, self.typ, self.random_seed, self.sparse_factor =  Rinfo_bucket.get_info()
        self.total_num = np.prod(self.tensor_shape)
        self.gen_typ = gen_typ
        self.noise_level = noise_level
        self.Tinfo_bucket = Tinfo_bucket
        self.Rinfo_bucket = Rinfo_bucket


    def ho_svd(self):
        X = square_tensor_gen(self.n, self.rank, dim=self.dim, typ=self.gen_typ, noise_level=self.noise_level)
        start_time = time.time()
        core, tucker_factors = tucker(X, ranks=[self.rank for _ in range(self.dim)], init='random')
        X_hat = tl.tucker_to_tensor(core, tucker_factors)
        running_time = time.time() - start_time
        rerr = eval_mse(X,X_hat)
        return (-1, running_time), rerr

    def two_pass(self):
        X = square_tensor_gen(self.n, self.rank, dim=self.dim, typ=self.gen_typ, noise_level=self.noise_level)
        start_time = time.time()
        sketch = Sketch(X, self.k, random_seed = None)
        sketchs, _, = sketch.get_sketchs()
        sketch_time = time.time() - start_time
        start_time = time.time()
        sketch_two_pass = SketchTwoPassRecover(X, sketchs, self.rank)
        X_hat,_,_ =  sketch_two_pass.recover()
        recover_time = time.time() - start_time
        rerr = eval_mse(X,X_hat)
        return (sketch_time, recover_time), rerr

    def one_pass(self,store_rm = True):
        X = square_tensor_gen(self.n, self.rank, dim=self.dim, typ=self.gen_typ, noise_level=self.noise_level)
        start_time = time.time()
        sketch = Sketch(X, self.k, s = self.s, random_seed=self.random_seed,store_rm = True)
        sketchs, core_sketch, = sketch.get_sketchs()
        sketch_time = time.time() - start_time
        start_time = time.time()
        _, core_rm = sketch.get_rm()
        sketch_one_pass = SketchOnePassRecover(sketchs, core_sketch, self.Tinfo_bucket, self.Rinfo_bucket,core_rm = core_rm)
        X_hat,_,_ = sketch_one_pass.recover()
        recover_time = time.time() - start_time
        rerr = eval_mse(X,X_hat)
        return (sketch_time, recover_time), rerr

    def run(self, simu_typ, simu_runs):
        times = []
        rerrs = []
        rtime, rerr = None, None
        for i in range(simu_runs):
            if simu_typ == 'ho_svd':
                rtime, rerr = self.ho_svd()
            elif simu_typ == 'two_pass':
                rtime, rerr = self.two_pass()
            elif simu_typ == 'one_pass':
                rtime, rerr = self.one_pass()
            times.append(rtime)
            rerrs.append(rerr)
        print(times)
        return [sum(y) / len(y) for y in zip(*times)], np.mean(rerr)

if __name__ == '__main__':
    n = 200

    simu = Simulation(TensorInfoBucket([n,n,n], k = 12, rank = 5, s=80), \
        RandomInfoBucket(random_seed = 1), gen_typ = 'id', noise_level=0)

    rtime, rerr = simu.run(simu_typ='one_pass', simu_runs=10)
    print(rtime)
    print(rerr)








