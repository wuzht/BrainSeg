#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   brains18.py
@Time    :   2020/01/14 23:19:19
@Author  :   Wu
@Version :   1.0
@Desc    :   BrainS18 dataset
'''

# here put the import lib
from torch.utils.data import Dataset

import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import numpy as np
import os

class BrainS18Dataset(Dataset):
    def __init__(self, root_dir='./datasets/BrainS18', folders=['1', '5', '7', '4', '148', '070', '14']):
        # print('Preparing BrainS18Dataset {} ... '.format(folders), end='')
        self.file_names = ['_FLAIR.png', '_reg_IR.png', '_reg_T1.png', '_segm.png']
        self.mean_std = {}  # mean and std of a volume
        self.img_paths = [] # e.g. './datasets/BrainS18/14/2'
        self._prepare_dataset(root_dir, folders)

    def _prepare_dataset(self, root_dir, folders):
        # compute mean and std and prepare self.img_paths
        for folder in folders:
            paths = [os.path.join(root_dir, folder, str(i)) for i in range(48)]
            self.img_paths += paths
            self.mean_std[folder] = {}
            for file_name in ['_FLAIR.png', '_reg_IR.png', '_reg_T1.png']:
                volume = np.array([mpimg.imread(path + file_name) for path in paths])
                self.mean_std[folder][file_name] = [volume.mean(), volume.std()]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        folder = self.img_paths[index].split('/')[-2]
        # read imgs
        imgs = [mpimg.imread(self.img_paths[index] + fn).reshape((1, 240, 240)) for fn in self.file_names]
        # imgs[3]是标签图像
        imgs[3] *= 255
        # 标签图像必须是LongTensor
        imgs[3] = torch.LongTensor(imgs[3].reshape(240, 240))
        # normalization
        for i in range(3):
            # e.g. mean_std = {'_FLAIR.png': [0.14819147, 0.22584382], 
            #                  '_reg_IR.png': [0.740661, 0.18219014], 
            #                  '_reg_T1.png': [0.1633398, 0.25954548]}
            mean = self.mean_std[folder][self.file_names[i]][0]
            std = self.mean_std[folder][self.file_names[i]][1]
            imgs[i] = (imgs[i] - mean) / std

        return imgs

    def show_imgs(self, index):
        imgs = self.__getitem__(index)
        fig, axs = plt.subplots(2,2, sharey=True, figsize=(10,8))
        axs[0][0].set_title(self.file_names[0])
        axs[0][1].set_title(self.file_names[1])
        axs[1][0].set_title(self.file_names[2])
        axs[1][1].set_title(self.file_names[3])
        
        ax00 = axs[0][0].imshow(imgs[0][0], aspect="auto", cmap="gray")
        ax01 = axs[0][1].imshow(imgs[1][0], aspect="auto", cmap="gray")
        ax10 = axs[1][0].imshow(imgs[2][0], aspect="auto", cmap="gray")
        ax11 = axs[1][1].imshow(imgs[3].numpy(), aspect="auto", cmap=plt.cm.get_cmap('tab10', 10), vmin=0, vmax=9)

        fig.colorbar(ax00, ax=axs[0][0])
        fig.colorbar(ax01, ax=axs[0][1])
        fig.colorbar(ax10, ax=axs[1][0])
        fig.colorbar(ax11, ax=axs[1][1])

        fig.suptitle(self.img_paths[index])
        plt.show()


if __name__ == "__main__": 
    data = BrainS18Dataset()
    print(len(data))