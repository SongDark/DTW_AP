# coding:utf-8
import os, time 
import multiprocessing
import numpy as np 
from fastdtw import fastdtw 

l2_norm = lambda x, y: np.sqrt(np.sum(np.square(x-y)))

def dtw_dist(x, y, radius=10):
    return fastdtw(x, y, radius, l2_norm)[0]

def Worker(x, y, dist, m, n, k):
    mat = np.zeros((len(x), len(y)))
    if m==n:
        for i in range(len(x)):
            for j in range(i, len(y)):
                mat[i, j] = dist(x[i], y[j])
                mat[j, i] = mat[i, j]
    else:
        for i in range(len(x)):
            for j in range(len(y)):
                mat[i, j] = dist(x[i], y[j])
    
    # np.save("tmp{}.npy".format(m*k+n), mat)
    return mat 

def np_dtw_parallel(data, lens, dist=dtw_dist, num_of_block=25):
    # for 16-core CPU, 25 is upperbound
    k = int(np.sqrt(num_of_block))
    width_of_block = len(data) // k

    pool = multiprocessing.Pool(processes=(k + 1)*k//2) # 15, cannot larger than 16
    results = []
    for i in range(k):
        x = data[width_of_block*i:width_of_block*i+width_of_block]
        for j in range(i, k):
            y = data[width_of_block*j:width_of_block*j+width_of_block]
            results.append(pool.apply_async(Worker, (x,y, dist, i, j, k)))
    if width_of_block*i+width_of_block < len(data):
        x = data[width_of_block*i+width_of_block:]
        y = data
        results.append(pool.apply_async(Worker, (x,y, dist, i, j, k)))
    
    tmp = []
    for x in results:
        tmp.append(x.get())
    
    dist_mat = np.zeros((len(data), len(data)))
    cnt = 0
    for i in range(k):
        for j in range(i, k):
            mat = tmp[cnt]
            dist_mat[len(mat) * i : len(mat) * (i+1), len(mat) * j : len(mat) * (j+1)] = mat 
            cnt += 1
    for i in range(len(dist_mat)):
        for j in range(0, i):
            dist_mat[i][j] = dist_mat[j][i]
    return dist_mat
