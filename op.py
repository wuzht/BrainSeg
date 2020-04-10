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

from unet import DropoutUNet
from dice_loss import dice_coeff
from brains18 import BrainS18Dataset
from tools import ImProgressBar, save_model, load_model, save_model_all, load_model_all, Logger

def arr2str(arr):
    s = '['
    for x in arr:
        s += '{:.3f}, '.format(x)
    return s[:-2] + ']'


class Operation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.init_environment()

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
        self.model = DropoutUNet(n_channels=1, n_classes=cfg.n_classes, model_type=cfg.model_type, drop_rate=cfg.drop_rate)

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

    def init_environment(self):
        # 创建文件夹
        if not os.path.exists(self.cfg.exp_dir):
            os.makedirs(self.cfg.exp_dir)
        if not os.path.exists(self.cfg.cur_dir):
            os.makedirs(self.cfg.cur_dir)

        # Logger对象
        self.cfg.log = Logger(self.cfg.log_path, level='debug').logger
        self.cfg.log.info('config:\n{}'.format(self.cfg.cfg2str(self.cfg)))


    def load_val_data(self, is_tumor=True):
        self.val_data = BrainS18Dataset(folders=self.cfg.val_folds, is_tumor=is_tumor)

    def load(self, path, mode=True):
        """
        mode: 加载模型方式
            True: 加载模型参数
            False: 加载完整模型
        """
        self.model = DropoutUNet(n_channels=1, n_classes=self.cfg.n_classes, model_type=self.cfg.model_type, drop_rate=self.cfg.drop_rate)
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


    def train(self, data_loader, is_dropout=True):
        total_loss = 0
        self.model.train()
        pbar = ImProgressBar(len(data_loader))
        # train the model using minibatch
        num = 0
        for i, (imgs, _, _) in enumerate(data_loader):
            num += imgs[2].shape[0]
            batch_x = imgs[2].to(self.device)
            batch_y = imgs[3].to(self.device)

            # forward
            batch_y_pred = self.model(batch_x, is_dropout=is_dropout)
            loss = self.criterion(batch_y_pred, batch_y)

            # backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.update(i)
        pbar.finish()
        return total_loss / num

    def eval_model(self, data_loader, is_dropout=False):
        # 评价单个模型
        with torch.no_grad():
            pbar = ImProgressBar(len(data_loader))
            self.model.eval()

            total_loss = 0
            # dices for each class
            total_dices = np.zeros(self.cfg.n_classes)
            num = 0
            for i, (imgs, _, slice_id) in enumerate(data_loader):
                num += imgs[2].shape[0]
                batch_x = imgs[2].to(self.device)
                batch_y = imgs[3].to(self.device)

                out = self.model(batch_x, is_dropout=is_dropout)
                loss = self.criterion(out, batch_y)
                total_loss += loss.item()
                
                _, batch_y_pred = torch.max(out, dim=1)

                cur_dices = self.get_dices(batch_y_pred, batch_y)

                total_dices += cur_dices
                pbar.update(i)
            pbar.finish()
            return total_dices / num, total_loss / num


    def eval_model_dices(self, data, is_dropout=False):
        # 单个模型的dices
        N, B = self.cfg.n_classes, 1
        C, H, W = data[0][0][2].shape

        pbar = ImProgressBar(len(data))
        self.model.eval()

        # dices for each class
        total_dices = np.zeros(self.cfg.n_classes)
        partitioned_dices = np.zeros(shape=(3, self.cfg.n_classes))     # 分块dices, 分下中上层，0-15下，16-31中，32-47上

        with torch.no_grad():
            for i in range(len(data)):
                imgs, _, slice_id = data[i]
                x = torch.from_numpy(imgs[2]).to(self.device).reshape(B, C, H, W) # B,C,H,W
                y_gt = imgs[3].to(self.device).reshape(B, H, W)

                output = self.model(x, is_dropout=is_dropout)
                _, y_pred = torch.max(output, dim=1)
                dices = self.get_dices(y_pred, y_gt)
                total_dices += dices
                partitioned_dices[int(slice_id/16)] += dices
                pbar.update(i)
            pbar.finish()

            total_dices /= (i+1)
            partitioned_dices *= 3 / (i+1)
            self.print_dices(total_dices, partitioned_dices)
            return total_dices, partitioned_dices

    
    def eval_sample_model_dices(self, data, sample, is_dropout=True):
        # 采样模型或集成模型的dices
        N, B = self.cfg.n_classes, 1
        C, H, W = data[0][0][2].shape
        T = self.cfg.sample_T if sample else len(self.models)
        pbar = ImProgressBar(len(data))
        self.model.eval()

        # dices for each class
        total_dices = np.zeros(self.cfg.n_classes)
        partitioned_dices = np.zeros(shape=(3, self.cfg.n_classes))     # 分块dices, 分下中上层，0-15下，16-31中，32-47上

        with torch.no_grad():
            for i in range(len(data)):
                imgs, _, slice_id = data[i]
                x = torch.from_numpy(imgs[2]).to(self.device).reshape(B, C, H, W) # B,C,H,W
                y_gt = imgs[3].to(self.device).reshape(B, H, W)
                results = torch.zeros(T, N, H, W).to(self.device)               # (18, 9, 240, 240) 18 是模型数

                # # 预测T次
                if sample:
                    for j in range(T):
                        output = self.model(x, is_dropout=is_dropout)
                        results[j] = F.softmax(output, dim=1)
                else:
                    for j, model in enumerate(self.models):
                        output = model(x)           # B,N,H,W           
                        results[j] = F.softmax(output, dim=1)

                p = torch.mean(results, dim=0, keepdim=True)            # B,N,H,W
                _, y_pred = torch.max(p, dim=1)
                dices = self.get_dices(y_pred, y_gt)
                total_dices += dices
                partitioned_dices[int(slice_id/16)] += dices
                pbar.update(i)
            pbar.finish()

            total_dices /= (i+1)
            partitioned_dices *= 3 / (i+1)
            self.print_dices(total_dices, partitioned_dices)
            return total_dices, partitioned_dices


    def print_dices(self, total_dices, partitioned_dices):
        self.cfg.log.info("")
        self.cfg.log.info("Class     : {} [c1-c8 mean]".format(['{:3d}'.format(x) for x in range(0, self.cfg.n_classes)]))
        self.cfg.log.info("Total dice: {} [{:.3f}]".format(arr2str(total_dices), total_dices[1:].mean()))
        self.cfg.log.info("Bottomdice: {} [{:.3f}]".format(arr2str(partitioned_dices[0]), partitioned_dices[0][1:].mean()))
        self.cfg.log.info("Mid   dice: {} [{:.3f}]".format(arr2str(partitioned_dices[1]), partitioned_dices[1][1:].mean()))
        self.cfg.log.info("Up    dice: {} [{:.3f}]".format(arr2str(partitioned_dices[2]), partitioned_dices[2][1:].mean()))


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

        best_epoch = 0
        best_val_dice = 0

        for epoch in range(self.cfg.epochs):
            self.cfg.log.info('[Epoch {}/{}]'.format(epoch + 1, self.cfg.epochs))

            loss = self.train(self.train_loader, is_dropout=True)
            
            train_dices, train_loss = self.eval_model(self.train_loader, is_dropout=False)
            val_dices, val_loss = self.eval_model(self.val_loader, is_dropout=False)

            self.cfg.log.info("Train Loss: {:.6f} (from train)".format(loss))
            self.cfg.log.info("Train Loss: {:.6f} (from eval )".format(train_loss))
            self.cfg.log.info("Val   Loss: {:.6f} (from eval )".format(val_loss))
            
            self.cfg.log.info("Class     : {} [mDice c1-8]".format(['{:3d}'.format(x) for x in range(0, self.cfg.n_classes)]))
            self.cfg.log.info("Train dice: {} [{:.3f}]".format(arr2str(train_dices), train_dices[1:].mean()))
            self.cfg.log.info("Val   dice: {} [{:.3f}]".format(arr2str(val_dices), val_dices[1:].mean()))

            self.save(self.cfg.model_path, mode=True)
            # self.save(self.cfg.model_all_path, mode=False)
            if val_dices[1:].mean() > best_val_dice:
                best_epoch = epoch + 1
                best_val_dice = val_dices[1:].mean()
                self.save(self.cfg.model_best_path, mode=True)

            self.cfg.log.info("best val dice: [{:.3f}] (best epoch: {})".format(best_val_dice, best_epoch))
        self.cfg.log.critical('Train finished')


    def predict(self, image, y_gt, title="", is_dropout=False):
        """
        N: n_classes        (e.g. 9)
        B: batch_size       (e.g. 10)
        C: image channels   (e.g. 1)
        H: image height     (e.g. 240)
        W: image width      (e.g. 240)
        """
        N, B = self.cfg.n_classes, 1
        C, H, W = image.shape
        self.model.eval()

        with torch.no_grad():
            x = torch.from_numpy(image).to(self.device).reshape(B, C, H, W) # B,C,H,W
            y_gt = y_gt.to(self.device).reshape(B, H, W)

            output = self.model(x, is_dropout=is_dropout)
            p = F.softmax(output, dim=1).cpu().data.numpy()
            entropy = -np.sum(p * np.log(p), axis=1) # B,H,W

            _, y_pred = torch.max(output, dim=1)

            dices = self.get_dices(y_pred, y_gt)

            # 转numpy
            y_pred = y_pred.cpu().data.numpy()[0]
            y_gt = y_gt.cpu().data.numpy()[0]

            # show me the result
            fig, axs = plt.subplots(2,2, sharey=True, figsize=(10,8))
            axs[0][0].set_title("Original data")
            axs[0][1].set_title("Ground Truth")
            axs[1][0].set_title("Entropy [{:.3f}, {:.3f}]".format(entropy.min(), entropy.max()))
            axs[1][1].set_title("Prediction")
            plt.suptitle("({}) dice : {} [c1-c8 mean: {:.3f}]".format(title, arr2str(dices), dices[1:].mean()))
            
            # cmap = plt.cm.get_cmap('Paired', 10)    # 10 discrete colors
            cmap = plt.cm.get_cmap('tab10', 10)    # 10 discrete colors
            # cmap = plt.cm.get_cmap('Set3', 10)    # 10 discrete colors

            ax00 = axs[0][0].imshow( image[0,...], aspect="auto")
            ax01 = axs[0][1].imshow( y_gt, cmap=cmap, aspect="auto", vmin=0, vmax=9)
            ax10 = axs[1][0].imshow( entropy[0,...],  aspect="auto", cmap=plt.cm.get_cmap('jet'), vmin=0, vmax=2)
            ax11 = axs[1][1].imshow( y_pred, cmap=cmap, aspect="auto", vmin=0, vmax=9)
            
            fig.colorbar(ax00, ax=axs[0][0])
            fig.colorbar(ax01, ax=axs[0][1])
            fig.colorbar(ax10, ax=axs[1][0])
            fig.colorbar(ax11, ax=axs[1][1])


    def predict_sample(self, image, y_gt, title="", sample=True, is_dropout=True):
        """
        N: n_classes        (e.g. 9)
        B: batch_size       (e.g. 10)
        C: image channels   (e.g. 1)
        H: image height     (e.g. 240)
        W: image width      (e.g. 240)
        T: number of ensemble models or number of samples   (e.g. 18)
        """
        N = self.cfg.n_classes
        B = 1
        C, H, W = image.shape
        T = self.cfg.sample_T if sample else len(self.models)
        self.model.eval()

        with torch.no_grad():
            x = torch.from_numpy(image).to(self.device).reshape(B, C, H, W) # B,C,H,W
            y_gt = y_gt.to(self.device).reshape(B, H, W)                    # B,H,W
            results = torch.zeros(T, N, H, W).to(self.device)               # (18, 9, 240, 240) 18 是模型数

            # 预测T次
            if sample:
                for i in range(T):
                    output = self.model(x, is_dropout=is_dropout)
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

            self.save_figs(image, y_gt, entropy, y_pred, variance, title, dices)
            # self.show_fig(image, y_gt, entropy, y_pred, variance, title, dices)
            

    def show_fig(self, image, y_gt, entropy, y_pred, variance, title, dices):
        # show me the result
        fig, axs = plt.subplots(2,3, sharey=True, figsize=(16,8.5))
        axs[0][0].set_title("Original data")
        axs[1][0].set_title("Ground Truth")
        axs[0][1].set_title("Entropy [{:.3f}, {:.3f}]".format(entropy.min(), entropy.max()))
        axs[1][1].set_title("Prediction")
        axs[0][2].set_title("Variance [{:.3f}, {:.3f}]".format(variance.min(), variance.max()))
        plt.suptitle("({}) dice : {} [c1-c8 mean: {:.3f}]".format(title, arr2str(dices), dices[1:].mean()))
        
        # cmap = plt.cm.get_cmap('Paired', 10)    # 10 discrete colors
        cmap = plt.cm.get_cmap('tab10', 10)    # 10 discrete colors
        # cmap = plt.cm.get_cmap('Set3', 10)    # 10 discrete colors

        ax00 = axs[0][0].imshow( image[0], aspect="auto", cmap='gray')
        ax10 = axs[1][0].imshow( y_gt, cmap=cmap, aspect="auto", vmin=0, vmax=9)
        ax01 = axs[0][1].imshow( entropy,  aspect="auto", cmap=plt.cm.get_cmap('jet'), vmin=0.0, vmax=2.0)
        ax11 = axs[1][1].imshow( y_pred, cmap=cmap, aspect="auto", vmin=0, vmax=9)
        ax02 = axs[0][2].imshow( variance, aspect="auto", cmap=plt.cm.get_cmap('jet'), vmin=0.0, vmax=0.35)
        
        fig.colorbar(ax00, ax=axs[0][0])
        fig.colorbar(ax10, ax=axs[1][0])
        fig.colorbar(ax01, ax=axs[0][1])
        fig.colorbar(ax11, ax=axs[1][1])
        fig.colorbar(ax02, ax=axs[0][2])


    def save_figs(self, image, y_gt, entropy, y_pred, variance, title, dices):
        # 创建文件夹
        if not os.path.exists(self.cfg.result_dir):
            os.makedirs(self.cfg.result_dir)

        # show me the result
        fig, axs = plt.subplots(nrows=1,ncols=5, sharey=True, figsize=(21,4))
        fig.tight_layout() # 调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0) # 调整子图间距
        axs[0].set_title("Original data")
        axs[1].set_title("Ground Truth")
        axs[3].set_title("Entropy [{:.3f}, {:.3f}]".format(entropy.min(), entropy.max()))
        axs[2].set_title("Prediction")
        axs[4].set_title("Variance [{:.3f}, {:.3f}]".format(variance.min(), variance.max()))
        # plt.suptitle("({}) dice : {} [c1-c8 mean: {:.3f}]".format(title, arr2str(dices), dices[1:].mean()))
        for i in range(5):
            axs[i].axis('off') 

        # cmap = plt.cm.get_cmap('Paired', 10)    # 10 discrete colors
        cmap = plt.cm.get_cmap('tab10', 10)    # 10 discrete colors
        # cmap = plt.cm.get_cmap('Set3', 10)    # 10 discrete colors

        ax00 = axs[0].imshow( image[0], aspect="auto", cmap='gray')
        ax10 = axs[1].imshow( y_gt, cmap=cmap, aspect="auto", vmin=0, vmax=9)
        ax01 = axs[3].imshow( entropy,  aspect="auto", cmap=plt.cm.get_cmap('jet'), vmin=0.0, vmax=2.0)
        ax11 = axs[2].imshow( y_pred, cmap=cmap, aspect="auto", vmin=0, vmax=9)
        ax02 = axs[4].imshow( variance, aspect="auto", cmap=plt.cm.get_cmap('jet'), vmin=0.0, vmax=0.33)
        
        fig.colorbar(ax00, ax=axs[0])
        fig.colorbar(ax10, ax=axs[1])
        fig.colorbar(ax01, ax=axs[3])
        fig.colorbar(ax11, ax=axs[2])
        fig.colorbar(ax02, ax=axs[4])

        name = title
        plt.savefig(os.path.join(self.cfg.result_dir, name))

    
    def ensemble_load_models(self):
        model_paths = [
            'D-No-brains18=0409-160305',
            'D-No-brains18=0409-160500',
            'D-No-brains18=0409-160529',
            'D-No-brains18=0409-160600',
            'D-No-brains18=0409-160621',
            'D-No-brains18=0409-160701',
            'D-No-brains18=0409-165858',
            'D-No-brains18=0409-165911',
            'D-No-brains18=0409-165924',
            'D-No-brains18=0409-165939',
            'D-No-brains18=0409-165954',
            'D-No-brains18=0409-170009',
            'D-No-brains18=0409-182832',
            'D-No-brains18=0409-182843',
            'D-No-brains18=0409-182855',
            'D-No-brains18=0409-182906',
            'D-No-brains18=0409-182919',
            'D-No-brains18=0409-182933',
            'D-No-brains18=0409-192735',
            'D-No-brains18=0409-192747',
            'D-No-brains18=0409-192753',
            'D-No-brains18=0409-192804',
            'D-No-brains18=0409-192816',
            'D-No-brains18=0409-192831',
            'D-No-brains18=0409-201646',
            'D-No-brains18=0409-201702',
            'D-No-brains18=0409-201705',
            'D-No-brains18=0409-201722',
            'D-No-brains18=0409-201738',
            'D-No-brains18=0409-201752'
        ] # 30
        model_paths = ['exp/{}/model_best.pt'.format(x) for x in model_paths]

        self.models = []
        for path in model_paths:
            model = DropoutUNet(n_channels=1, n_classes=self.cfg.n_classes, model_type='No')
            model = load_model(model, path, self.device)
            model.eval()
            self.models.append(model)
            self.cfg.log.info("Model loaded from {}".format(path)) 
        self.cfg.log.info("T (number of models): {}".format(len(self.models)))


    def rm_dir(self):
        os.system('rm -rf {}'.format(self.cfg.cur_dir))
