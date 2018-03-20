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
        mse,rmse = eval_mse(X,X_hat)
        return mse, running_time,rmse 

    def simu_two_pass(self):
        X = square_tensor_gen(self.n, self.rank, dim=self.dim, typ=self.gen_typ, noise_level=self.noise_level)
        start_time = time.time()
        sketch = Sketch(X, self.k, random_seed = None)
        sketchs, _, = sketch.get_sketchs()
        sketch_time = time.time() - start_time
        start_time = time.time()
        sketch_two_pass = SketchTwoPassRecover(X, sketchs, self.rank)
        X_hat,_,_ =  sketch_two_pass.recover()
        recover_time = time.time() - start_time
        mse,rmse = eval_mse(X,X_hat)
        return mse, sketch_time, recover_time,rmse

    def simu_one_pass(self):
        X = square_tensor_gen(self.n, self.rank, dim=self.dim, typ=self.gen_typ, noise_level=self.noise_level)
        start_time = time.time()
        sketch = Sketch(X, self.k, s = self.s, random_seed=self.random_seed)
        sketchs, core_sketch, = sketch.get_sketchs()
        sketch_time = time.time() - start_time
        start_time = time.time()
        sketch_one_pass = SketchOnePassRecover(sketchs, core_sketch, self.Tinfo_bucket, self.Rinfo_bucket)
        X_hat,_,_ = sketch_one_pass.recover()
        recover_time = time.time() - start_time
        mse,rmse= eval_mse(X,X_hat) 
        return mse, sketch_time, recover_time,rmse 

    def run(self, simu_typ, simu_runs):
        running_times = []
        mse_arr = []
        rmse_arr = []
        if simu_typ == 'ho_svd':
            for i in range(simu_runs):
                mse, running_time,rmse = self.ho_svd()
                running_times.append((-1, running_time))
                mse_arr.append(mse)
                rmse_arr.append(rmse)

        if simu_typ == 'two_sketch':
            for i in range(simu_runs):
                mse, sketch_time, recover_time,rmse = self.simu_two_pass()
                mse_arr.append(mse)
                running_times.append((sketch_time, recover_time))
                rmse_arr.append(rmse)
        if simu_typ == 'one_sketch':
            for i in range(simu_runs):
                mse, sketch_time, recover_time,rmse = self.simu_one_pass()
                mse_arr.append(mse)
                running_times.append((sketch_time, recover_time))
                rmse_arr.append(rmse)
        return running_times, mse_arr,rmse_arr

if __name__ == '__main__':
    n = 200

    simu = Simulation(TensorInfoBucket([n,n,n], k = 15, rank = 10, s=30), \
        RandomInfoBucket(random_seed = 1), gen_typ = 'id', noise_level=0.1)


    running_times, mse_arr,rmse_arr = simu.run(simu_typ = 'ho_svd', simu_runs = 10)
    print(running_times)
    print(mse_arr)


    running_times, mse_arr,rmse_arr = simu.run(simu_typ='two_sketch', simu_runs=10)
    print(running_times)
    print(mse_arr)

    running_times, mse_arr,rmse_arr = simu.run(simu_typ='one_sketch', simu_runs=10)
    print(running_times)
    print(mse_arr)








