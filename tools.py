# coding: utf-8

import logging
from logging import handlers
import sys
import os
import numpy as np


def choose_gpu(gpu_not_use=[]):
    """
    return the id of the gpu with the most memory
    """
    # query GPU memory and save the result in `tmp`
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    # read the file `tmp` to get a gpu memory list
    memory_gpu = [int(x.split()[2]) for x in open('tmp','r').readlines()]

    for i in gpu_not_use:
        memory_gpu[i] = 0   # not use these gpus

    # get the id of the gpu with the most memory
    gpu_id = str(np.argmax(memory_gpu))
    # remove the file `tmp`
    os.system('rm tmp')

    # msg = 'memory_gpu: {}'.format(memory_gpu)
    return gpu_id, memory_gpu


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(
            filename=filename, when=when, backupCount=backCount, encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒、M 分、H 小时、D 天、W 每星期（interval==0时代表星期一）、midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


class ImProgressBar(object):
    def __init__(self, total_iter, bar_len=50):
        self.total_iter = total_iter
        self.bar_len = bar_len
        self.coef = self.bar_len / 100
        self.foo = ['-', '\\', '|', '/']

    def update(self, i):
        sys.stdout.write('\r')
        progress = int((i + 1) / self.total_iter * 100)
        sys.stdout.write("[%4s/%4s] %3s%% |%s%s| %s" % (
            (i + 1),
            self.total_iter,
            progress,
            int(progress * self.coef) * '>',
            (self.bar_len - int(progress * self.coef)) * ' ',
            self.foo[(i + 1) % len(self.foo)]
        ))
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write('\n')


import torch

def save_model(model, path):
    """
    Save the model `model` to `path`\\
    Args: 
        path: The path of the model to be saved
        model: The model to save
    """
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    """
    Load `model` from `path`, and push `model` to `device`\\
    Args: 
        model: The model to save
        path: The path of the model to be saved
        device: the torch device
    Return:
        the loaded model
    """
    saved_params = torch.load(path)
    model.load_state_dict(saved_params)
    model = model.to(device)
    return model


def save_model_all(model, path):
    torch.save(model, path)


def load_model_all(path, device):
    return torch.load(path).to(device)
