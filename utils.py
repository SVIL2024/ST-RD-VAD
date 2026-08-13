import os
import random
import numpy as np
import torch


def setup_seed(seed):
    """固定所有随机种子，保证实验可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write2txt(log_dir, content):
    """将一行日志追加写入 log_dir/log.txt。"""
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'log.txt'), 'a', encoding='utf-8') as f:
        f.write(str(content) + "\n")