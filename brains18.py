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

import numpy as np
import nibabel
from scipy import ndimage
import os
from matplotlib.pyplot import imshow
from skimage import exposure
import skimage.io as io


def prepare_data(root_dir="./datasets/BrainS18/BrainS18", dst_dir='./datasets/BrainS18/normal', folders=['1', '5', '7', '4', '148', '070', '14']):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    filenames = ['FLAIR.nii.gz', 'reg_T1.nii.gz', 'reg_IR.nii.gz']
    for folder in folders:
        for filename in filenames:
            full_src_path = os.path.join(root_dir, folder, 'pre', filename)
            volume = nibabel.load(full_src_path).get_data()

            num_of_image = volume.shape[2]
            new_folder = os.path.join(dst_dir, folder)
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)

            for i in range(num_of_image):
                new_filename = '{}_{}.png'.format(str(i), filename[:-7])
                slice = volume[:,:,i]
                slice = exposure.equalize_adapthist(slice.astype(np.int))
                print(np.max(slice), np.min(slice))
                io.imsave(os.path.join(new_folder, new_filename), np.rot90(slice))        
            print(full_src_path)

    # 标签图像
    print("----------------------")
    filenames = ['segm.nii.gz']
    for folder in folders:
        for filename in filenames:
            full_src_path = os.path.join(root_dir, folder, filename)
            volume = nibabel.load( full_src_path ).get_data()
            num_of_image = volume.shape[2]
            new_folder = os.path.join(dst_dir, folder)
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)
            for i in range(num_of_image):
                new_filename = '{}_{}.png'.format(str(i), filename[:-7])
                io.imsave(os.path.join(new_folder, new_filename), np.rot90(volume[:,:,i]) )
            print(full_src_path)


class BrainS18Dataset(Dataset):
    def __init__(self, 
    root_dir='./datasets/BrainS18/normal', 
    folders=['1', '5', '7', '4', '148', '070', '14'],
    file_names=['_FLAIR.png', '_reg_IR.png', '_reg_T1.png', '_segm.png'],
    is_tumor=False):
        if is_tumor:
            folders = ['{}_Brats17_CBICA_AAB_1_img'.format(x) for x in folders]
            file_names = ['_FLAIR.png', '_orgin.png', '_reg_T1.png', '_segm.png']
        self.is_tumor = is_tumor
        # print('Preparing BrainS18Dataset {} ... '.format(folders), end='')
        self.file_names = file_names
        self.mean_std = {}  # mean and std of a volume
        self.img_paths = [] # e.g. './datasets/BrainS18/14/2'
        self._compute_mean_std(root_dir, folders)

    def _compute_mean_std(self, root_dir, folders):
        # compute mean and std and prepare self.img_paths
        for folder in folders:
            paths = [os.path.join(root_dir, folder, str(i)) for i in range(48)]
            self.img_paths += paths
            self.mean_std[folder] = {}
            for file_name in self.file_names[:3]:
                volume = np.array([mpimg.imread(path + file_name) for path in paths])
                self.mean_std[folder][file_name] = [volume.mean(), volume.std()]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        folder = self.img_paths[index].split('/')[-2]
        slice_id = int(self.img_paths[index].split('/')[-1])

        # read imgs
        if self.is_tumor:
            # 旋转180度
            imgs = [np.rot90(mpimg.imread(self.img_paths[index] + fn), 2).reshape((1, 240, 240)) for fn in self.file_names]
        else:
            imgs = [mpimg.imread(self.img_paths[index] + fn).reshape((1, 240, 240)) for fn in self.file_names]

        imgs[3] *= 255                  # imgs[3]是标签图像
        imgs[3][imgs[3] == 9] = 1       # 将第9类归为第1类
        imgs[3] = torch.LongTensor(imgs[3].copy().reshape(240, 240))   # 标签图像必须是LongTensor
        
        # normalization
        for i in range(3):
            # e.g. mean_std = {'_FLAIR.png': [0.14819147, 0.22584382], 
            #                  '_reg_IR.png': [0.740661, 0.18219014], 
            #                  '_reg_T1.png': [0.1633398, 0.25954548]}
            mean = self.mean_std[folder][self.file_names[i]][0]
            std = self.mean_std[folder][self.file_names[i]][1]
            imgs[i] = (imgs[i] - mean) / std

        return imgs, folder, slice_id

    def show_imgs(self, index):
        imgs, _, _ = self.__getitem__(index)
        fig, axs = plt.subplots(2,2, sharey=True, figsize=(10,8))
        axs[0][0].set_title(self.file_names[0])
        axs[0][1].set_title(self.file_names[1])
        axs[1][0].set_title(self.file_names[2])
        axs[1][1].set_title(self.file_names[3])
        
        ax00 = axs[0][0].imshow(imgs[0][0], aspect="auto", cmap="gray", interpolation='none')
        ax01 = axs[0][1].imshow(imgs[1][0], aspect="auto", cmap="gray", interpolation='none')
        ax10 = axs[1][0].imshow(imgs[2][0], aspect="auto", cmap="gray", interpolation='none')
        ax11 = axs[1][1].imshow(imgs[3].numpy(), aspect="auto", cmap=plt.cm.get_cmap('tab10', 10), vmin=0, vmax=9, interpolation='none')

        fig.colorbar(ax00, ax=axs[0][0])
        fig.colorbar(ax01, ax=axs[0][1])
        fig.colorbar(ax10, ax=axs[1][0])
        fig.colorbar(ax11, ax=axs[1][1])

        fig.suptitle(self.img_paths[index])
        plt.show()

    def get_class_weight(self, n_classes=9):
        class_num = np.zeros(n_classes)   # 统计每类的像素点个数
        img_num = self.__len__()
        for i in range(img_num):
            imgs, _, _ = self.__getitem__(i)
            for c in range(n_classes):
                class_num[c] += np.sum(imgs[3].numpy() == c)

        class_weight = 1.0 / class_num
        sum_class_weight = np.sum(class_weight)
        class_weight = [x / sum_class_weight for x in class_weight]

        return class_weight

    

if __name__ == "__main__": 
    # data = BrainS18Dataset()
    # print(len(data))
    prepare_data()