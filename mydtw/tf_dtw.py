# coding:utf-8
import tensorflow as tf
import numpy as np

'''
    tf.reduce_sum will cause an strange Error when batchsize is large,
    this Error makes two positive value adding up to a negative value,
    which makes tf.sqrt return 'nan'.
'''
def tf_l2norm_distmat_batch(X, Y, data_format='NWC'):
    '''
        X: [x1, x2, x3] (N,T,d) Y: [y1, y2, y3] (N,T,d)
        compute l2_norm
        return: [D(x1,y1), D(x2,y2), D(x3,y3)] (T*T,N)
    '''
    if data_format == 'WNC':
        X = tf.transpose(X, (1,0,2))
        Y = tf.transpose(Y, (1,0,2))
        T, N, d = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
    else:
        N, T, d = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]

    X = tf.reshape(tf.tile(X, [1, 1, T]), (N*T*T, d))
    Y = tf.reshape(tf.tile(Y, [1, T, 1]), (N*T*T, d))
    
    # res = tf.sqrt(tf.reduce_sum(tf.squared_difference(X, Y), 1)) # it would return nans !!
    res = tf.squared_difference(X, Y)
    res = tf.sqrt(res[:,0] + res[:, 1] + res[:, 2]) # assume C=3

    res = tf.reshape(res, [N, T, T]) # [N, Tx, Ty]
    res = tf.transpose(res, (1,2,0)) # [Tx, Ty, N]
    res = tf.reshape(res, [T*T, N]) # [Tx*Ty, N]
    return res

def tf_dtw_batch(X, Y, data_format='NWC'):
    '''
        X: [x1, x2, x3] (N,T,d) Y: [y1, y2, y3] (N,T,d)
        returns the accumulated matrix D
        D[len_x, len_y] is the DTW distance between x and y
    '''

    # X, Y : [N, T, d]
    dist_mats = tf_l2norm_distmat_batch(X, Y, data_format=data_format) # [T*T, N]
    batch_size, max_time = tf.shape(X)[0], tf.shape(X)[1]

    d_array = tf.TensorArray(tf.float32, size=max_time*max_time, clear_after_read=False)
    d_array = d_array.unstack(dist_mats) # read(t) returns an [N,] array at t timestep

    D = tf.TensorArray(tf.float32, size=(max_time+1)*(max_time+1), clear_after_read=False)
    
    # initalize
    def cond_boder(idx, res):
        return idx < max_time+1
    def body_border_x(idx, res):
        res = res.write(tf.to_int32(idx * (max_time+1)), 10000*tf.ones(shape=(batch_size, )))
        return idx+1, res
    def body_border_y(idx, res):
        res = res.write(tf.to_int32(idx), 10000*tf.ones(shape=(batch_size, )))
        return idx+1, res
    _, D = tf.while_loop(cond_boder, body_border_x, (1, D)) 
    _, D = tf.while_loop(cond_boder, body_border_y, (1, D)) 

    def cond(idx, res):
        return idx < (max_time+1) * (max_time+1)
    def body(idx, res):
        i = tf.to_int32(tf.divide(idx, max_time+1))
        j = tf.mod(tf.to_int32(idx), max_time+1)
        def f1():
            dt = d_array.read(i*(max_time+1)+j-max_time-i-1)
            min_v = tf.minimum(res.read((i - 1) * (max_time + 1) + j), res.read(i * (max_time + 1) + j - 1))
            min_v = tf.minimum(min_v, res.read((i - 1) * (max_time + 1) + j - 1))
            return res.write(idx, min_v + dt)
        def f2():
            return res
        res = tf.cond(tf.less(i, 1) | tf.less(j, 1), 
            true_fn=f2,
            false_fn=f1) 
        return idx+1, res
    _, D = tf.while_loop(cond, body, (0, D)) 

    D = D.stack()
    D = tf.reshape(D, (max_time+1, max_time+1, batch_size)) # [T+1, T+1, N]
    
    return D


def tf_dtw(dataset, lens, batch_size=None):
    '''
        dataset [N, maxlen, d]
    '''
    N, T, d = dataset.shape 
    batch_size = batch_size or (N-1)*N//2
    batch_size = min(batch_size, (N-1)*N//2, 10000)

    idx_is, idx_js = [], []
    for i in range(N):
        for j in range(i+1, N):
            idx_is.append(i)
            idx_js.append(j)
    print N, len(idx_is)
    for _ in range(batch_size - len(idx_is) + len(idx_is)//batch_size * batch_size):
        idx_is.append(0)
        idx_js.append(0)
    idx_is, idx_js = np.array(idx_is), np.array(idx_js)
    print N, len(idx_is), batch_size

    X = tf.placeholder(dtype=tf.float32, shape=(batch_size, T, d))
    Y = tf.placeholder(dtype=tf.float32, shape=(batch_size, T, d))
    D = tf_dtw_batch(X, Y)

    with tf.Session() as sess:
        res = []
        for i in range(len(idx_is)//batch_size):
            print "batch {}".format(i)
            cur_range = range(i*batch_size, (i+1)*batch_size)
            cur_is, cur_js = idx_is[cur_range], idx_js[cur_range]
            feed_dict = {
                X : dataset[cur_is],
                Y : dataset[cur_js]
            }
            cur_D = sess.run(D, feed_dict=feed_dict) # [T+1, T+1, bz]
            res.append([cur_D[lens[cur_is[j]], lens[cur_js[j]], j] for j in range(batch_size)])
    res = np.concatenate(res)

    dtw_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1, N):
            dtw_mat[i,j] = res[i*N+j - (i+1)*(i+2)//2]
            dtw_mat[j,i] = dtw_mat[i,j]
    
    return dtw_mat