# coding:utf-8

import numpy as np 
import pandas 
import arff 
import os

class CharacterTrajectories_Processor():
    def __init__(self):
        self.data_path = "/home/scw4750/songbinxu/datasets/CharacterTrajectories/"
        # self.convert_arff_to_npz()
    
    def convert_arff_to_npz(self):
        npz_data = {}
        for group in ['TRAIN', 'TEST']:
            data, label = [], []
            for i in range(1, 4):
                arff_data = arff.load(os.path.join(self.data_path, "CharacterTrajectoriesDimension{}_".format(i) + group + ".arff"))
                arff_data = np.array(list(arff_data)).astype(np.float32)
                data.append(np.expand_dims(np.cumsum(arff_data[:, :-1], 1), -1))
                label.append(arff_data[:, -1])
            assert (np.sum(label[0] - label[1]) == 0) and (np.sum(label[1] - label[2]) == 0)
            data = np.concatenate(data, -1)
            print data.shape, label[0].shape
            npz_data['x_'+group.lower()] = data 
            npz_data['y_'+group.lower()] = (label[0] - 1).astype(np.int32) # 1~20 -> 0~19
        np.savez(os.path.join(self.data_path, "CharacterTrajectories.npz"), **npz_data)

# cp = CharacterTrajectories_Processor()
# cp.convert_arff_to_npz()

