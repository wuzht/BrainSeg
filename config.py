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
parser.add_argument("--epochs", type=int, default=256, help='迭代次数')
parser.add_argument("--lr", type=float, default=1e-3, help='学习率')
parser.add_argument("--weight_decay", type=float, default=5e-4, help='权重衰减')
parser.add_argument("--momentum", type=float, default=0.9, help='动量')
parser.add_argument("--epsilon", type=float, default=1e-8)
parser.add_argument("--dropout", type=bool, default=True, help='是否使用Dropout模型')
parser.add_argument("--is_class_weight", type=bool, default=False, help='是否应用class weight')
parser.add_argument("--model_type", type=str, default='Mid1', 
    choices=['All','Encoder','Decoder','Center1','Center2','Mid1','Mid1-Encoder','Mid1-Decoder','Classifier','No'],
    help='DropoutUnet的类型')
parser.add_argument("--drop_rate", type=float, default=0.2, help='Dropout probability')
parser.add_argument("--sample_T", type=int, default=50, help='采样T次')

# Data
parser.add_argument("--is_test", type=bool, default=False)
parser.add_argument("--data_name", type=str, default='brains18', help='数据集名称')
parser.add_argument("--n_classes", type=int, default=9, help='标签数（背景是第0类,第9类归为第1类）')
parser.add_argument("--batch_size", type=int, default=16, help='批大小')
parser.add_argument("--num_workers", type=int, default=8, help='线程数')
parser.add_argument("--folders", type=list, default=['1', '5', '7', '4', '148', '070', '14'], help='文件夹')
parser.add_argument("--is_tumor", type=bool, default=False, help='是否肿瘤数据')
parser.add_argument("--val_folds", type=list, default=['1'], help='验证集文件夹')

# Arguments
cfg = parser.parse_args(args=[]) # jupyter运行
# cfg = parser.parse_args()        # 命令行运行

# Data paths
cfg.train_folds = [x for x in cfg.folders if x not in cfg.val_folds]     # 训练集文件夹

# Device
gpu_id, cfg.memory_gpu = tools.choose_gpu(gpu_not_use=[0])
cfg.device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

# Paths
if cfg.is_test:
    cfg.cur_dir = os.path.join(cfg.exp_dir, 'test-{}={}'.format(cfg.data_name, now_time))    # 当前实验记录文件夹
else:
    cfg.cur_dir = os.path.join(cfg.exp_dir, '{}-{}{}={}'.format(
        'D' if cfg.dropout else 'E',
        (cfg.model_type + '-') if cfg.dropout else '',
        cfg.data_name,
        now_time)
    )    # 当前实验记录文件夹
cfg.model_path = os.path.join(cfg.cur_dir, 'model.pt')                              # 模型参数路径(保存模型参数)
cfg.model_best_path = os.path.join(cfg.cur_dir, 'model_best.pt')                    # 模型参数路径(保存模型参数,val集最佳)
cfg.model_all_path = os.path.join(cfg.cur_dir, 'model.pth')                         # 模型路径(保存完整模型)
cfg.log_path = os.path.join(cfg.cur_dir, 'exp.log')                                 # log文件路径

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

cfg.print_cfg = print_cfg
cfg.cfg2str = cfg2str