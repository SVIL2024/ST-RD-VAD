import os
import random
import numpy as np
import torch
import json
from collections import defaultdict


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


def save_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def list_video_names(video_folder):
    names = sorted(
        name for name in os.listdir(video_folder)
        if os.path.isdir(os.path.join(video_folder, name))
    )
    if not names:
        raise RuntimeError("No video directories found in {}".format(video_folder))
    return names


def _split_one_group(names, val_ratio, rng):
    names = sorted(names)
    rng.shuffle(names)
    if len(names) <= 1:
        return names, []
    val_count = max(1, min(int(round(len(names) * val_ratio)), len(names) - 1))
    return sorted(names[val_count:]), sorted(names[:val_count])


def split_train_val_videos(video_names, val_ratio=0.2, seed=2026, dataset_type="ped2"):
    """Deterministic video-level split; ShanghaiTech is stratified by scene."""
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if dataset_type.lower() != "shanghai":
        return _split_one_group(video_names, val_ratio, random.Random(seed))

    groups = defaultdict(list)
    for name in video_names:
        groups[name.split("_", 1)[0]].append(name)
    train_names, val_names = [], []
    for offset, scene in enumerate(sorted(groups)):
        train_group, val_group = _split_one_group(
            groups[scene], val_ratio, random.Random(seed + offset)
        )
        train_names.extend(train_group)
        val_names.extend(val_group)
    return sorted(train_names), sorted(val_names)
