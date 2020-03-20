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

from unet import UNet, DropoutUNet
from dice_loss import dice_coeff
from brains18 import BrainS18Dataset
from tools import ImProgressBar, save_model, load_model, save_model_all, load_model_all

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

        # Data loaders
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
        if not cfg.dropout:
            self.model = UNet(n_channels=1, n_classes=cfg.n_classes)
        else:
            self.model = DropoutUNet(n_channels=1, n_classes=cfg.n_classes)

        # Criterion
        if cfg.is_class_weight:
            self.class_weight = self.train_data.get_class_weight()
            self.cfg.log.critical("class_weight: \n{}".format(self.class_weight))
            self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.class_weight))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.SGD(self.model.parameters(),
                            lr           = cfg.lr,
                            momentum     = cfg.momentum,
                            weight_decay = cfg.weight_decay
                            )
        
        self.cfg.log.critical("criterion: \n{}".format(self.criterion))
        self.cfg.log.critical("optimizer: \n{}".format(self.optimizer))
        self.cfg.log.critical("model: \n{}".format(self.model))


    def load(self, path, mode=True):
        """
        mode: 加载模型方式
            True: 加载模型参数
            False: 加载完整模型
        """
        if mode:
            self.model = load_model(self.model, path, self.device)
            self.cfg.log.info("Model param loaded from {}".format(path))
        else:
            self.model = load_model_all(path, self.device)
            self.cfg.log.info("Model loaded from {}".format(path))


    def save(self, path, mode=True):
        """
        mode: 加载模型方式
            True: 加载模型参数
            False: 加载完整模型
        """
        if mode:
            save_model(self.model, path)
            self.cfg.log.info("Model param saved at {}".format(path))
        else:
            save_model_all(self.model, path)
            self.cfg.log.info("Model saved at {}".format(path))


    def train(self, data_loader):
        total_loss = 0
        self.model.train()
        pbar = ImProgressBar(len(data_loader))
        # train the model using minibatch
        for i, (imgs, _, _) in enumerate(data_loader):
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
        with torch.no_grad():
            self.model.eval()
            if self.cfg.dropout:
                self.model.train()

            total_loss = 0
            # dices for each class
            total_dices = np.zeros(self.cfg.n_classes)
            for i, (imgs, _, _) in enumerate(data_loader):
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

            self.save(self.cfg.model_path, mode=True)
            self.save(self.cfg.model_path, mode=False)

        self.cfg.log.critical('Train finished')


    def predict(self, image, y_gt):
        with torch.no_grad():
            self.model.eval()
            if self.cfg.dropout:
                self.model.train()
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
            y_pred = y_pred.cpu().data.numpy()
            y_gt = y_gt.cpu().data.numpy()

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


    def predict_sample(self, image, y_gt, title="", sample=False):
        """
        N: n_classes        (e.g. 9)
        B: batch_size       (e.g. 10)
        C: image channels   (e.g. 1)
        H: image height     (e.g. 240)
        W: image width      (e.g. 240)
        M: number of ensemble models or number of samples   (e.g. 18)
        """
        N = self.cfg.n_classes
        B = 1
        C, H, W = image.shape
        M = len(self.models) if mode == "ensemble" else 18

        with torch.no_grad():
            x = torch.from_numpy(image).to(self.device).reshape(B, C, H, W) # B,C,H,W
            y_gt = y_gt.to(self.device).reshape(B, H, W)                    # B,H,W
            results = torch.zeros(M, N, H, W).to(self.device)               # (18, 9, 240, 240) 18 是模型数

            # 预测M次
            if sample:
                for i in range(M):
                    output = self.model(x)
                    results[i] = F.softmax(output, dim=1)
            else:
                for i, model in enumerate(self.models):
                    output = model(x)           # B,N,H,W           
                    results[i] = F.softmax(output, dim=1)
                

            # 得到预测结果和dice
            p = torch.mean(results, dim=0, keepdim=True)            # B,N,H,W
            _, y_pred = torch.max(p, dim=1)
            dices = self.get_dices(y_pred, y_gt)

            # 计算variance
            variance = np.var(results.cpu().data.numpy(), axis=0)   # N,H,W
            variance = np.sum(variance, axis=0)                     # H,W

            # 计算entropy
            p_numpy = p.cpu().data.numpy()[0]                       # N,H,W
            entropy = -np.sum(p_numpy * np.log(p_numpy), axis=0)    # H,W  这里的乘法 * 是对应位置逐点相乘

            # 转numpy
            y_pred = y_pred.cpu().data.numpy()[0]
            y_gt = y_gt.cpu().data.numpy()[0]

            self.show_fig(image, y_gt, entropy, y_pred, variance, title, dices)


    def show_fig(self, image, y_gt, entropy, y_pred, variance, title, dices):
        # show me the result
        fig, axs = plt.subplots(2,3, sharey=True, figsize=(16,8.5))
        axs[0][0].set_title("Original data")
        axs[1][0].set_title("Ground Truth")
        axs[0][1].set_title("Entropy [{:.3f}, {:.3f}]".format(entropy.min(), entropy.max()))
        axs[1][1].set_title("Prediction")
        axs[0][2].set_title("Variance [{:.3f}, {:.3f}]".format(variance.min(), variance.max()))
        plt.suptitle("({}) dice : {}".format(title, arr2str(dices)))
        
        # cmap = plt.cm.get_cmap('Paired', 10)    # 10 discrete colors
        cmap = plt.cm.get_cmap('tab10', 10)    # 10 discrete colors
        # cmap = plt.cm.get_cmap('Set3', 10)    # 10 discrete colors

        ax00 = axs[0][0].imshow( image[0], aspect="auto")
        ax10 = axs[1][0].imshow( y_gt, cmap=cmap, aspect="auto", vmin=0, vmax=9)
        ax01 = axs[0][1].imshow( entropy,  aspect="auto", cmap=plt.cm.get_cmap('jet'), vmin=0.0, vmax=2.0)
        ax11 = axs[1][1].imshow( y_pred, cmap=cmap, aspect="auto", vmin=0, vmax=9)
        ax02 = axs[0][2].imshow( variance, aspect="auto", cmap=plt.cm.get_cmap('jet'), vmin=0.0, vmax=0.5)
        
        fig.colorbar(ax00, ax=axs[0][0])
        fig.colorbar(ax10, ax=axs[1][0])
        fig.colorbar(ax01, ax=axs[0][1])
        fig.colorbar(ax11, ax=axs[1][1])
        fig.colorbar(ax02, ax=axs[0][2])

    
    def ensemble_load_models(self):
        model_paths = [
            'E-brains18=0314-180735', 
            'E-brains18=0314-182316', 
            'E-brains18=0314-184617', 
            'E-brains18=0314-184900', 
            'E-brains18=0314-191031', 
            'E-brains18=0314-191616', 
            'E-brains18=0314-193439', 
            'E-brains18=0314-194709', 
            'E-brains18=0314-195238', 
            'E-brains18=0314-200317', 
            'E-brains18=0314-201357', 
            'E-brains18=0314-203550', 
            'E-brains18=0314-205210', 
            'E-brains18=0314-212917', 
            'E-brains18=0314-213325', 
            'E-brains18=0314-214800', 
            'E-brains18=0314-220258', 
            'E-brains18=0314-221005'
        ]
        model_paths = ['exp/{}/model.pt'.format(x) for x in model_paths]

        self.models = []
        for path in model_paths:
            model = UNet(n_channels=1, n_classes=self.cfg.n_classes)
            model = load_model(model, path, self.device)
            model.eval()
            self.models.append(model)
            print("Model loaded from {}".format(path)) 
        print("M (number of models): {}".format(len(self.models)))


    
        