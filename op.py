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
from tools import ImProgressBar, save_model, load_model, Logger
from viewer import Viewer, arr2str


class Operation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        torch.cuda.set_device(self.device)
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

    def rm_dir(self):
        os.system('rm -rf {}'.format(self.cfg.cur_dir))

    def load_val_data(self, is_tumor=True):
        self.val_data = BrainS18Dataset(folders=self.cfg.val_folds, is_tumor=is_tumor)

    def load(self, path, model_type):
        self.model = DropoutUNet(n_channels=1, n_classes=self.cfg.n_classes, model_type=model_type, drop_rate=self.cfg.drop_rate)
        self.model = load_model(self.model, path, self.device)
        self.cfg.log.info("Model param loaded from {}".format(path))

    def save(self, path):
        save_model(self.model, path)
        self.cfg.log.info("Model param saved at {}".format(path))


############################################################################################################################
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
        """评价单个模型,训练时使用"""
        with torch.no_grad():
            pbar = ImProgressBar(len(data_loader))
            self.model.eval()

            total_loss = 0
            # dices for each class
            total_dices = np.zeros(self.cfg.n_classes)
            num = 0
            for i, (imgs, _, _) in enumerate(data_loader):
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

            self.save(self.cfg.model_path)
            if val_dices[1:].mean() > best_val_dice:
                best_epoch = epoch + 1
                best_val_dice = val_dices[1:].mean()
                self.save(self.cfg.model_best_path)

            self.cfg.log.info("best val dice: [{:.3f}] (best epoch: {})".format(best_val_dice, best_epoch))
        self.cfg.log.critical('Train finished')

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


############################################################################################################################
    def eval_dices(self, data, mode):
        assert(mode == 'Dropout' or mode == 'Ensemble' or mode == 'Simple')
        N, B = self.cfg.n_classes, 1
        C, H, W = data[0][0][2].shape
        is_dropout = (mode != 'Simple')
        self.model.eval()
        pbar = ImProgressBar(len(data))

        # dices for each class
        total_dices = np.zeros(self.cfg.n_classes)
        partitioned_dices = np.zeros(shape=(3, self.cfg.n_classes))     # 分块dices, 分下中上层，0-15下，16-31中，32-47上
        
        with torch.no_grad():
            for i in range(len(data)):
                imgs, _, slice_id = data[i]
                x = torch.from_numpy(imgs[2]).to(self.device).reshape(B, C, H, W) # B,C,H,W
                y_gt = imgs[3].to(self.device).reshape(B, H, W)

                if mode == 'Simple':
                    output = self.model(x, is_dropout=is_dropout)
                else:
                    T = self.cfg.sample_T if mode == 'Dropout' else len(self.models)
                    results = torch.zeros(T, N, H, W).to(self.device)               # (18, 9, 240, 240) 18 是模型数
                    # 预测T次
                    if mode == 'Dropout':
                        for j in range(T):
                            output = self.model(x, is_dropout=is_dropout)
                            results[j] = F.softmax(output, dim=1)
                    else:   # mode == 'Ensemble'
                        for j, model in enumerate(self.models):
                            output = model(x)           # B,N,H,W           
                            results[j] = F.softmax(output, dim=1)
                    output = torch.mean(results, dim=0, keepdim=True)            # B,N,H,W

                _, y_pred = torch.max(output, dim=1)
                dices = self.get_dices(y_pred, y_gt)
                total_dices += dices
                partitioned_dices[int(slice_id/16)] += dices
                pbar.update(i)
            pbar.finish()

            total_dices /= (i+1)
            partitioned_dices *= 3 / (i+1)
            Viewer.print_dices(self.cfg.log.info, total_dices, partitioned_dices)
            return total_dices, partitioned_dices


    def inference(self, image, y_gt, mode):
        """
        N: n_classes        (e.g. 9)
        B: batch_size       (e.g. 10)
        C: image channels   (e.g. 1)
        H: image height     (e.g. 240)
        W: image width      (e.g. 240)
        T: number of ensemble models or number of samples   (e.g. 30)
        """
        assert(mode == 'Dropout' or mode == 'Ensemble' or mode == 'Simple')
        N, B = self.cfg.n_classes, 1
        C, H, W = image.shape
        is_dropout = (mode != 'Simple')
        self.model.eval()

        x = torch.from_numpy(image).to(self.device).reshape(B, C, H, W) # B,C,H,W
        y_gt = y_gt.to(self.device).reshape(B, H, W)                    # B,H,W

        if mode == 'Simple':
            with torch.no_grad():
                output = self.model(x, is_dropout=is_dropout)
                p_numpy = F.softmax(output, dim=1).cpu().data.numpy()
                entropy = -np.sum(p_numpy * np.log(p_numpy), axis=1) # B,H,W
                _, y_pred = torch.max(output, dim=1)
                dices = self.get_dices(y_pred, y_gt)

                # 转numpy
                y_pred = y_pred.cpu().data.numpy()[0]
                # y_gt = y_gt.cpu().data.numpy()[0]

                return y_pred, entropy, dices
        else:
            T = self.cfg.sample_T if mode == 'Dropout' else len(self.models)
            with torch.no_grad():
                results = torch.zeros(T, N, H, W).to(self.device)               # (T, 9, 240, 240)

                # 预测T次
                if mode == 'Dropout':
                    for i in range(T):
                        output = self.model(x, is_dropout=is_dropout)
                        results[i] = F.softmax(output, dim=1)
                else:   # mode == 'Ensemble'
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
                # y_gt = y_gt.cpu().data.numpy()[0]
                
                return y_pred, entropy, variance, dices


    def predict(self, image, y_gt, title=""):
        """
        N: n_classes        (e.g. 9)
        B: batch_size       (e.g. 10)
        C: image channels   (e.g. 1)
        H: image height     (e.g. 240)
        W: image width      (e.g. 240)
        """
        y_pred, entropy, dices = self.inference(image, y_gt, mode='Simple')
        Viewer.show_fig_4(image, y_gt, entropy, y_pred, title, dices)

    def predict_sample(self, image, y_gt, mode, title=""):
        
        assert(mode == 'Dropout' or mode == 'Ensemble')
        y_pred, entropy, variance, dices = self.inference(image, y_gt, mode=mode)
        # Viewer.save_figs(self.cfg.result_dir, image, y_gt, entropy, y_pred, variance, title, dices)
        Viewer.show_fig(image, y_gt, entropy, y_pred, variance, title, dices)


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

    
    def do_sth(self, y_gt, entropy, variance):
        label_count = np.zeros(self.cfg.n_classes)
        entropy_sum = np.zeros(self.cfg.n_classes)
        variance_sum = np.zeros(self.cfg.n_classes)

        for c in range(self.cfg.n_classes):
            label_count[c] += np.sum(y_gt == c)
            entropy_sum[c] += np.sum(entropy[y_gt == c])
            variance_sum[c] += np.sum(variance_sum[y_gt == c])
        
        return label_count, entropy_sum, variance_sum

    # def do_more(self)

