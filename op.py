import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import numpy as np
import os

import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
import torch.nn.functional as F

from dropout_unet import UNet
from dice_loss import dice_coeff
from brains18 import BrainS18Dataset
from tools import ImProgressBar, save_model, load_model


def arr2str(arr):
    s = '['
    for x in arr:
        s += '{:.3f}, '.format(x)
    return s[:-2] + ']'


class Operation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        # Data
        self.train_data = BrainS18Dataset(folders=cfg.train_folds)
        self.val_data = BrainS18Dataset(folders=cfg.val_folds)

        self.train_loader = Data.DataLoader(
            dataset=self.train_data,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers
        )
        self.val_loader = Data.DataLoader(
            dataset=self.val_data,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers
        )

        # Model
        self.model = UNet(n_channels=1, n_classes=cfg.n_classes, dropout=cfg.dropout)

        if self.cfg.is_class_weight:
            self.class_weight = self.train_data.get_class_weight()
            self.cfg.log.critical("class_weight: \n{}".format(self.class_weight))
            self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.class_weight))
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(),
                            lr           = cfg.lr,
                            momentum     = cfg.momentum,
                            weight_decay = cfg.weight_decay
                            )
        
        self.cfg.log.critical("criterion: \n{}".format(self.criterion))
        self.cfg.log.critical("optimizer: \n{}".format(self.optimizer))
        self.cfg.log.critical("model: \n{}".format(self.model))


    def load(self, path):
        self.model = load_model(self.model, path, self.device)
        self.cfg.log.info("Model loaded from {}".format(path))


    def train(self, data_loader):
        total_loss = 0
        self.model.train()
        pbar = ImProgressBar(len(data_loader))
        # train the model using minibatch
        for i, imgs in enumerate(data_loader):
            batch_x = imgs[2].to(self.device)
            batch_y = imgs[3].to(self.device)

            # forward
            batch_y_pred = self.model(batch_x)
            loss = self.criterion(batch_y_pred, batch_y)

            # backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.update(i)
        pbar.finish()
        return total_loss / len(data_loader)


    def eval_model(self, data_loader):
        """Evaluation without the densecrf with the dice coefficient"""
        self.model.eval()

        total_loss = 0
        # dices for each class
        total_dices = np.zeros(self.cfg.n_classes)
        for i, imgs in enumerate(data_loader):
            batch_x = imgs[2].to(self.device)
            batch_y = imgs[3].to(self.device)

            out = self.model(batch_x)
            loss = self.criterion(out, batch_y)
            total_loss += loss.item()
            
            _, batch_y_pred = torch.max(out, dim=1)

            total_dices += self.get_dices(batch_y_pred, batch_y)
        return total_dices / (i+1), total_loss / (i+1)


    def get_dices(self, y_pred, y_gt):
        """
        y_pred: 预测得到的mask
        y_gt: Ground Truth
        """
        dices = np.zeros(self.cfg.n_classes)
        for c in range(self.cfg.n_classes):
            _x = (y_pred == c).float()
            _y = (y_gt == c).float()
            dices[c] += dice_coeff(_x, _y, self.device).item()
        return dices


    def fit(self):
        self.model.to(self.device)
        self.criterion.to(self.device)

        self.cfg.log.critical('Start training')
        for epoch in range(self.cfg.epochs):
            self.cfg.log.info('[Epoch {}/{}]'.format(epoch + 1, self.cfg.epochs))

            loss = self.train(self.train_loader)
            

            train_dices, train_loss = self.eval_model(self.train_loader)
            val_dices, val_loss = self.eval_model(self.val_loader)

            self.cfg.log.info("Train Loss: {:.4f}".format(train_loss))
            self.cfg.log.info("Val   Loss: {:.4f}".format(val_loss))
            
            self.cfg.log.info("Class     : {}".format(['{:3d}'.format(x) for x in range(0, self.cfg.n_classes)]))
            self.cfg.log.info("Train dice: {}".format(arr2str(train_dices)))
            self.cfg.log.info("Val   dice: {}".format(arr2str(val_dices)))

            save_model(self.model, self.cfg.model_path)
            self.cfg.log.info("Model saved at {}".format(self.cfg.model_path))

        self.cfg.log.critical('Train finished')


    def predict(self, image, y_gt):
        with torch.no_grad():
            self.model.eval()
            x = torch.from_numpy(image).to(self.device)
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])    # B,C,H,W

            y_gt = y_gt.to(self.device)
            y_gt = y_gt.reshape(1, y_gt.shape[0], y_gt.shape[1])    # B,H,W

            output = self.model(x)
            p = F.softmax(output, dim=1).cpu().data.numpy()
            entropy = -np.sum(p * np.log(p), axis=1) # (shape: (batch_size, h, w))

            _, y_pred = torch.max(output, dim=1)

            dices = self.get_dices(y_pred, y_gt)

            # 转numpy
            y_pred = y_pred.cpu().data.numpy() if torch.cuda.is_available() else y_pred.data.numpy()
            y_gt = y_gt.cpu().data.numpy() if torch.cuda.is_available() else y_gt.data.numpy()


            # show me the result
            fig, axs = plt.subplots(2,2, sharey=True, figsize=(10,8))
            axs[0][0].set_title("Original data")
            axs[0][1].set_title("Ground Truth")
            axs[1][0].set_title("Entropy [{:.3f}, {:.3f}]".format(entropy.min(), entropy.max()))
            axs[1][1].set_title("Prediction")
            plt.suptitle("dice : {}".format(arr2str(dices)))
            
            # cmap = plt.cm.get_cmap('Paired', 10)    # 10 discrete colors
            cmap = plt.cm.get_cmap('tab10', 10)    # 10 discrete colors
            # cmap = plt.cm.get_cmap('Set3', 10)    # 10 discrete colors

            ax00 = axs[0][0].imshow( image[0,...], aspect="auto")
            ax01 = axs[0][1].imshow( y_gt[0], cmap=cmap, aspect="auto", vmin=0, vmax=9)
            ax10 = axs[1][0].imshow( entropy[0,...],  aspect="auto", cmap=plt.cm.get_cmap('jet'), vmin=0, vmax=2)
            ax11 = axs[1][1].imshow( y_pred[0,...], cmap=cmap, aspect="auto", vmin=0, vmax=9)
            
            fig.colorbar(ax00, ax=axs[0][0])
            fig.colorbar(ax01, ax=axs[0][1])
            fig.colorbar(ax10, ax=axs[1][0])
            fig.colorbar(ax11, ax=axs[1][1])
