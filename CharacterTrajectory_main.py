# coding:utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np 
import time
from sklearn.cluster import AffinityPropagation
from mydtw.tf_dtw import tf_dtw
from mydtw.np_dtw import np_dtw
from mydtw.np_dtw_parallel import np_dtw_parallel

def compute_dtw_tf(data, N=2858, batch=408):
    
    dataset = np.concatenate([data['x_train'], data['x_test']])
    dataset = dataset[:N]
    lens = len(dataset) * [182]

    start = time.time()
    dtw_mat_tf = tf_dtw(dataset, lens, batch_size=N*(N-1)/2//batch)
    print 'tf cost {} sec.'.format(time.time() - start)

    np.save('save/CharacterTrajectories/CharacterTrajectories_dtwmat_tf.npy', dtw_mat_tf)

    plt.imshow(dtw_mat_tf.T, origin='lower', cmap='gray', interpolation='nearest')
    plt.savefig('save/CharacterTrajectories/CharacterTrajectories_dtwmat_tf.png')

    return dtw_mat_tf

def compute_dtw_np(N=50):
    data = np.load('datasets/CharacterTrajectories/CharacterTrajectories.npz')

    dataset = data['x_train'][:N]
    lens = len(dataset) * [182]

    start = time.time()
    dtw_mat_np = np_dtw(dataset, lens, radius=10)
    print 'cost {} sec.'.format(time.time() - start)

    np.save('save/CharacterTrajectories/CharacterTrajectories_dtwmat_np.npy', dtw_mat_np)

    plt.imshow(dtw_mat_np.T, origin='lower', cmap='gray', interpolation='nearest')
    plt.savefig('save/CharacterTrajectories/CharacterTrajectories_dtwmat_np.png')

    return dtw_mat_np

def compute_dtw_np_parallel(N=50):
    data = np.load('datasets/CharacterTrajectories/CharacterTrajectories.npz')

    dataset = data['x_train'][:N]
    lens = len(dataset) * [182]

    start = time.time()
    dtw_mat_np = np_dtw_parallel(dataset, lens)
    print 'cost {} sec.'.format(time.time() - start)

    np.save('save/CharacterTrajectories/CharacterTrajectories_dtwmat_np_parallel.npy', dtw_mat_np)

    plt.imshow(dtw_mat_np.T, origin='lower', cmap='gray', interpolation='nearest')
    plt.savefig('save/CharacterTrajectories/CharacterTrajectories_dtwmat_np_parallel.png')

    return dtw_mat_np

def CharacterTrajectories_Train_Test():
    data = np.load('datasets/CharacterTrajectories/CharacterTrajectories.npz')
    train_size, test_size = len(data['x_train']), len(data['x_test'])
    # compute DTW matrix
    # dtw_distmat = compute_dtw_tf(data, N=2858, batch=408)
    dtw_distmat = np.load('save/CharacterTrajectories/CharacterTrajectories_dtwmat_tf.npy')

    train_dtw_distmat = -1.0 * dtw_distmat[:train_size, :train_size]

    # AP clustering
    af = AffinityPropagation().fit(train_dtw_distmat)
    cluster_centers_indices, labels = af.cluster_centers_indices_, af.labels_
    
    train_preds = data['y_train'][cluster_centers_indices[np.argmin(dtw_distmat[cluster_centers_indices, :train_size], 0)]]
    test_preds = data['y_test'][cluster_centers_indices[np.argmin(dtw_distmat[cluster_centers_indices, train_size:], 0)]]
    train_acc= np.sum(np.equal(data['y_train'], train_preds).astype(np.int32)) / float(train_size)
    test_acc= np.sum(np.equal(data['y_test'], test_preds).astype(np.int32)) / float(test_size)
    
    print "train_acc=%.3f, test_acc=%.3f" % (train_acc, test_acc)

    confuse_matrix = np.zeros((20, 20))
    for i in range(test_size):
        confuse_matrix[data['y_test'][i], test_preds[i]] += 1.0
    confuse_matrix = 1.0-(confuse_matrix - np.min(confuse_matrix))/(np.max(confuse_matrix)-np.min(confuse_matrix))

    ax = plt.subplot(111)
    plt.imshow(confuse_matrix, origin='lower', cmap='gray', interpolation='nearest')
    labels = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'y', 'z']
    plt.xticks(range(20))
    plt.yticks(range(20))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.savefig('save/CharacterTrajectories/CharacterTrajectories_confusion_matrix.png')
    

CharacterTrajectories_Train_Test()
