import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker

class skechyTwoPassRecover(object):
    def __init__(self, X, sketchs, rank):
        tl.set_backend('numpy')
        Qs = []
        self.arms = []
        for sketch in sketchs:
            Q, _ = np.linalg.qr(sketch)
            Qs.append(Q)
        core_tensor = X
        N = len(X.shape)
        for mode_n in range(N):
            core_tensor = tl.tenalg.mode_dot(core_tensor, Q.T, mode=mode_n)
        core, factors =  tucker(core_tensor, ranks=[rank, rank, rank])
        self.core = core
        for n in range(len(factors)):
            self.arms.append(np.dot(Qs[n], factors[n]))
        X_hat = self.core
        for n in range(len(factors)):
            X_hat = tl.tenalg.mode_dot(core_tensor, self.arms[n], mode=n)
        dif = X-X_hat
        dif = np.norm(dif.reshape(np.size(dif),1), 'fro')
        return self.arms, self.core ,dif






