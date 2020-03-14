import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import numpy as np
import os

from tools import load_model, choose_gpu
from dropout_unet import UNet
from brains18 import BrainS18Dataset
from dice_loss import dice_coeff
from op import arr2str

class EnsembleModel():
    def __init__(self, model_paths):
        super().__init__()
        self.models = []
        self.model_paths = model_paths

        gpu_id, memory_gpu = choose_gpu()
        print(gpu_id, memory_gpu)
        self.device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()

        # config
        self.n_classes = 9
        self.batch_size = 10
        self.num_workers = 8

        self.folders = ['1', '5', '7', '4', '148', '070', '14']
        self.val_folds = ['1']
        self.train_folds = [x for x in self.folders if x not in self.val_folds]

        # Data
        self.train_data = BrainS18Dataset(folders=self.train_folds)
        self.val_data = BrainS18Dataset(folders=self.val_folds)

        self.train_loader = Data.DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        self.val_loader = Data.DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )



    def load_models(self):
        for path in self.model_paths:
            model = UNet(n_channels=1, n_classes=self.n_classes, dropout=False)
            model = load_model(model, path, self.device)
            model.eval()
            self.models.append(model)
            print("Model loaded from {}".format(path)) 
        print("M (number of models): {}".format(len(self.models)))


    def get_dices(self, y_pred, y_gt):
        """
        y_pred: 预测得到的mask
        y_gt: Ground Truth
        """
        dices = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            _x = (y_pred == c).float()
            _y = (y_gt == c).float()
            dices[c] += dice_coeff(_x, _y, self.device).item()
        return dices

    
    def predict(self, image, y_gt):
        M_float = float(len(self.models))

        with torch.no_grad():
            x = torch.from_numpy(image).to(self.device)
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])    # B,C,H,W

            y_gt = y_gt.to(self.device)
            y_gt = y_gt.reshape(1, y_gt.shape[0], y_gt.shape[1])    # B,H,W

            h, w = x.shape[2], x.shape[3]
            p = torch.zeros(1, self.n_classes, h, w).to(self.device)   # B,C,H,W

            for model in self.models:
                output = model(x)
                p_value = F.softmax(output, dim=1) # (shape: (batch_size, n_classes, h, w))
                p = p + p_value / M_float

            _, y_pred = torch.max(p, dim=1)

            p_numpy = p.cpu().data.numpy()

            # 这里的乘法 * 是对应位置逐点相乘
            entropy = -np.sum(p_numpy * np.log(p_numpy), axis=1) # (shape: (batch_size, h, w))

            dices = self.get_dices(y_pred, y_gt)

            # 转numpy
            y_pred = y_pred.cpu().data.numpy() if torch.cuda.is_available() else y_pred.data.numpy()
            y_gt = y_gt.cpu().data.numpy() if torch.cuda.is_available() else y_gt.data.numpy()

            # show me the result
            fig, axs = plt.subplots(2,2, sharey=True, figsize=(10,8))
            axs[0][0].set_title("Original data")
            axs[0][1].set_title("Ground Truth")
            axs[1][0].set_title("Entropy")
            axs[1][1].set_title("Prediction")
            plt.suptitle("dice : {}".format(arr2str(dices)))
            
            # cmap = plt.cm.get_cmap('Paired', 10)    # 10 discrete colors
            cmap = plt.cm.get_cmap('tab10', 10)    # 10 discrete colors
            # cmap = plt.cm.get_cmap('Set3', 10)    # 10 discrete colors

            ax00 = axs[0][0].imshow( image[0,...], aspect="auto")
            ax01 = axs[0][1].imshow( y_gt[0], cmap=cmap, aspect="auto", vmin=0, vmax=9)
            ax10 = axs[1][0].imshow( entropy[0,...],  aspect="auto", cmap=plt.cm.get_cmap('jet'))
            ax11 = axs[1][1].imshow( y_pred[0,...], cmap=cmap, aspect="auto", vmin=0, vmax=9)
            
            fig.colorbar(ax00, ax=axs[0][0])
            fig.colorbar(ax01, ax=axs[0][1])
            fig.colorbar(ax10, ax=axs[1][0])
            fig.colorbar(ax11, ax=axs[1][1])