# coding:utf-8
import numpy as np
from fastdtw import fastdtw 

l2_norm = lambda x, y: np.sqrt(np.sum(np.square(x-y)))

def np_dtw(dataset, lens, radius=20):
    N = dataset.shape[0]
    res = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            res[i,j], _ = fastdtw(dataset[i][:lens[i], :], dataset[j][:lens[j], :], radius=20, dist=l2_norm)
            res[j,i] = res[i,j]
    # [N, N]
    return res 