import argparse
import os
import tools
import torch
import datetime

now_time = datetime.datetime.strftime(datetime.datetime.now(), '%m%d-%H%M%S') # 当前时间

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--exp_dir", type=str, default='./exp', help='实验记录文件夹')

# Model

# Train
parser.add_argument("--epochs", type=int, default=128, help='迭代次数')
parser.add_argument("--lr", type=float, default=1e-4, help='学习率')
parser.add_argument("--weight_decay", type=float, default=5e-4, help='权重衰减')
parser.add_argument("--momentum", type=float, default=0.9, help='动量')
parser.add_argument("--epsilon", type=float, default=1e-8)
parser.add_argument("--dropout", type=bool, default=False)
parser.add_argument("--is_class_weight", type=bool, default=True)

# Data
parser.add_argument("--data_name", type=str, default='CW-brains18', help='数据集名称')
# parser.add_argument("--choose_class", type=int, default=4, help='（二分类）选择哪一类来分割，如标签为4的类')
parser.add_argument("--n_classes", type=int, default=9, help='标签数（背景是第0类）')
parser.add_argument("--batch_size", type=int, default=10, help='批大小')
parser.add_argument("--num_workers", type=int, default=8, help='线程数')
parser.add_argument("--folders", type=list, default=['1', '5', '7', '4', '148', '070', '14'], help='文件夹')
parser.add_argument("--val_folds", type=list, default=['1'], help='验证集文件夹')

# Arguments
cfg = parser.parse_args(args=[]) # jupyter运行
# cfg = parser.parse_args()        # 命令行运行

cfg.train_folds = [x for x in cfg.folders if x not in cfg.val_folds]     # 训练集文件夹

gpu_id, cfg.memory_gpu = tools.choose_gpu()
cfg.device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

cfg.cur_dir = os.path.join(cfg.exp_dir, '{}={}'.format(cfg.data_name, now_time))    # 当前实验记录文件夹
cfg.model_path = os.path.join(cfg.cur_dir, 'model.pt')                              # 模型路径
cfg.log_path = os.path.join(cfg.cur_dir, 'exp.log')                                 # log文件路径

# 创建文件夹
if not os.path.exists(cfg.exp_dir):
    os.makedirs(cfg.exp_dir)
if not os.path.exists(cfg.cur_dir):
    os.makedirs(cfg.cur_dir)

# Logger对象
cfg.log = tools.Logger(cfg.log_path, level='debug').logger

###########################

def get_cfg():
    return cfg

def print_cfg(cfg):
    for k, v in vars(cfg).items():
        print('{}: {}'.format(k, v))

def cfg2str(cfg):
    s = ''
    for k, v in vars(cfg).items():
        s += '{}: {}\n'.format(k, v)
    return s

cfg.log.info('config:\n{}'.format(cfg2str(cfg)))