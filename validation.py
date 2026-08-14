"""Training-only validation for the public ST-RD-VAD implementation.

The official normal training videos are split at video level. Validation uses
only held-out normal clips and deterministic temporally shuffled counterparts;
benchmark test videos and labels are never opened here.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from tqdm import tqdm

from data import Reconstruction3DDataLoader
from I3D_model import InceptionI3d
from loss import get_ano_map
from model import OcbeAndDecoder
from test import _build_fg_mask
from utils import list_video_names, save_json, setup_seed, split_train_val_videos


COSINE_MAP_WEIGHTS = (0.5, 1.0, 1.5)
MSE_MAP_WEIGHT = 1.0
TOPK_RATIO = 0.10
SMOOTH_SIGMA = 9.0
FOREGROUND_QUANTILE = 0.75
BACKGROUND_KEEP = 0.20
HARD_RATIO = 0.60
BACKGROUND_MIX = 0.70
SELECTION_TOLERANCE = 1e-8


def align_student_features(teacher_features, student_features):
    return tuple(
        student[:, : teacher.shape[1], ...]
        for teacher, student in zip(teacher_features, student_features)
    )


def per_sample_discrepancy(teacher_features, student_features):
    values = []
    for teacher, student in zip(teacher_features, student_features):
        distance = 1.0 - F.cosine_similarity(
            F.normalize(teacher, p=2, dim=1),
            F.normalize(student, p=2, dim=1),
            dim=1,
        )
        values.append(distance.flatten(1).mean(dim=1))
    return torch.stack(values).sum(dim=0)


def per_sample_mse(teacher_features, student_features):
    return torch.stack([
        (teacher - student).pow(2).flatten(1).mean(dim=1)
        for teacher, student in zip(teacher_features, student_features)
    ]).sum(dim=0)


def calculate_score_map(teacher_features, student_features, temporal_length):
    cosine_maps = []
    deep_mse = None
    for index, (teacher, student) in enumerate(zip(teacher_features, student_features)):
        cosine_map, _, mse_map = get_ano_map(teacher, student)
        cosine_map = F.interpolate(
            cosine_map, size=(temporal_length, 64, 64),
            mode="trilinear", align_corners=True,
        )[:, 0]
        cosine_maps.append(cosine_map)
        if index == 2:
            deep_mse = F.interpolate(
                mse_map, size=(temporal_length, 64, 64),
                mode="trilinear", align_corners=True,
            )[:, 0]

    def smooth(value):
        batch, temporal, height, width = value.shape
        value = value.reshape(batch * temporal, 1, height, width)
        value = F.avg_pool2d(value, kernel_size=3, stride=1, padding=1)
        return value.reshape(batch, temporal, height, width)

    cosine_maps = [smooth(value) for value in cosine_maps]
    deep_mse = smooth(deep_mse)
    score_map = sum(
        weight * value for weight, value in zip(COSINE_MAP_WEIGHTS, cosine_maps)
    )
    return score_map + MSE_MAP_WEIGHT * torch.log1p(deep_mse)


def _topk_mean(values, ratio=TOPK_RATIO):
    values = values.flatten()
    count = max(1, int(values.numel() * ratio))
    return torch.topk(values, count).values.mean()


def aggregate_scores(score_maps, images):
    results = []
    for batch_index in range(score_maps.shape[0]):
        image = images[batch_index: batch_index + 1]
        mask = _build_fg_mask(
            image, out_hw=64, q=FOREGROUND_QUANTILE,
            min_area_ratio=0.01, bg_mix=BACKGROUND_MIX,
        )
        frame_scores = []
        for frame_index in range(score_maps.shape[1]):
            foreground = mask[frame_index] > 0.5
            values = score_maps[batch_index, frame_index]
            hard = _topk_mean(values[foreground] if foreground.any() else values)
            soft_map = values * (
                BACKGROUND_KEEP + (1.0 - BACKGROUND_KEEP) * mask[frame_index]
            )
            soft = _topk_mean(soft_map)
            frame_scores.append(HARD_RATIO * hard + (1.0 - HARD_RATIO) * soft)
        results.append(torch.stack(frame_scores))
    return torch.stack(results)


def _sample_location(sample_path):
    path = Path(sample_path)
    return path.parent.name, max(0, int(path.stem) - 1)


def _update_scores(storage, coverage, video_name, start, values):
    end = min(start + len(values), len(storage[video_name]))
    if start < end:
        values = np.asarray(values[: end - start], dtype=np.float32)
        storage[video_name][start:end] = np.maximum(
            storage[video_name][start:end], values
        )
        coverage[video_name][start:end] = True


def separation_statistics(normal_scores, pseudo_scores):
    normal_scores = np.asarray(normal_scores, dtype=np.float64)
    pseudo_scores = np.asarray(pseudo_scores, dtype=np.float64)
    margin = float(pseudo_scores.mean() - normal_scores.mean())
    pooled_variance = 0.5 * (normal_scores.var() + pseudo_scores.var())
    separation = margin / max(float(np.sqrt(max(pooled_variance, 0.0))), 1e-12)
    return margin, float(separation)


@torch.no_grad()
def validate_model(encoder, decoder, val_loader, val_dataset, device, mse_weight=0.1):
    encoder.eval()
    decoder.eval()
    video_names = list(val_dataset.videos.keys())
    normal_scores = {
        name: np.zeros(val_dataset.videos[name]["length"], dtype=np.float32)
        for name in video_names
    }
    pseudo_scores = {name: np.zeros_like(normal_scores[name]) for name in video_names}
    coverage = {name: np.zeros_like(normal_scores[name], dtype=bool) for name in video_names}
    normal_objectives = []
    cursor = 0
    temporal_length = int(val_dataset._num_frames)

    for images, pseudo_images in tqdm(val_loader, desc="Validation", leave=False):
        images = images.to(device)
        pseudo_images = pseudo_images.to(device)
        teacher_normal = encoder(images)
        student_normal_raw = decoder(*teacher_normal)
        student_normal = align_student_features(teacher_normal, student_normal_raw[:3])
        teacher_pseudo = encoder(pseudo_images)
        student_pseudo_raw = decoder(*teacher_pseudo)
        student_pseudo = align_student_features(teacher_pseudo, student_pseudo_raw[:3])

        normal_objectives.extend((
            per_sample_discrepancy(teacher_normal, student_normal)
            + mse_weight * per_sample_mse(teacher_normal, student_normal)
        ).cpu().tolist())
        normal_clip_scores = aggregate_scores(
            calculate_score_map(teacher_normal, student_normal, temporal_length), images
        ).cpu().numpy()
        pseudo_clip_scores = aggregate_scores(
            calculate_score_map(teacher_pseudo, student_pseudo, temporal_length), pseudo_images
        ).cpu().numpy()

        for batch_index in range(images.shape[0]):
            video_name, start = _sample_location(val_dataset.samples[cursor + batch_index])
            _update_scores(normal_scores, coverage, video_name, start, normal_clip_scores[batch_index])
            _update_scores(pseudo_scores, coverage, video_name, start, pseudo_clip_scores[batch_index])
        cursor += int(images.shape[0])

    normal_all, pseudo_all = [], []
    for video_name in video_names:
        valid = coverage[video_name]
        normal = gaussian_filter(normal_scores[video_name], sigma=SMOOTH_SIGMA)
        pseudo = gaussian_filter(pseudo_scores[video_name], sigma=SMOOTH_SIGMA)
        normal_all.extend(normal[valid].tolist())
        pseudo_all.extend(pseudo[valid].tolist())
    labels = np.concatenate([np.zeros(len(normal_all)), np.ones(len(pseudo_all))])
    scores = np.asarray(normal_all + pseudo_all)
    margin, separation = separation_statistics(normal_all, pseudo_all)
    return {
        "aligned_scoring_auc": float(roc_auc_score(labels, scores)),
        "aligned_effect_size": separation,
        "aligned_score_margin": margin,
        "normal_objective": float(np.mean(normal_objectives)),
        "normal_frame_count": len(normal_all),
        "pseudo_frame_count": len(pseudo_all),
    }


def load_teacher(path, device):
    encoder = InceptionI3d(num_classes=400, in_channels=3, dropout_keep_prob=0.5)
    pretrained = torch.load(path, map_location=device)
    state = encoder.state_dict()
    compatible = {k: v for k, v in pretrained.items() if k in state and v.shape == state[k].shape}
    state.update(compatible)
    encoder.load_state_dict(state)
    return encoder.to(device).eval()


def is_better(candidate, incumbent):
    if incumbent is None:
        return True
    auc_delta = candidate["aligned_scoring_auc"] - incumbent["aligned_scoring_auc"]
    if auc_delta > SELECTION_TOLERANCE:
        return True
    if abs(auc_delta) > SELECTION_TOLERANCE:
        return False
    sep_delta = candidate["aligned_effect_size"] - incumbent["aligned_effect_size"]
    if sep_delta > SELECTION_TOLERANCE:
        return True
    if abs(sep_delta) > SELECTION_TOLERANCE:
        return False
    return candidate["normal_objective"] < incumbent["normal_objective"]


def dataset_paths(data_root, dataset_type):
    if dataset_type == "ped2":
        return os.path.join(data_root, "UCSDped2", "Train"), ".tif"
    if dataset_type == "avenue":
        return os.path.join(data_root, "Avenue", "Train"), ".jpg"
    return os.path.join(data_root, "ShanghaiTech", "training", "frames"), ".jpg"


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_folder, extension = dataset_paths(args.data_root, args.dataset_type)
    all_videos = list_video_names(train_folder)
    train_videos, val_videos = split_train_val_videos(
        all_videos, args.val_ratio, args.split_seed, args.dataset_type
    )
    dataset = Reconstruction3DDataLoader(
        train_folder, transforms.Compose([transforms.ToTensor()]),
        args.resize, args.resize, img_extension=extension,
        dataset=args.dataset_type, train=True, train_stride=args.val_stride,
        video_names=val_videos, pseudo_seed=args.val_pseudo_seed,
        deterministic_pseudo=True,
    )
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False)
    encoder = load_teacher(args.teacher_path, device)
    checkpoint_paths = []
    for pattern in args.checkpoints:
        checkpoint_paths.extend(sorted(glob.glob(pattern)))
    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoints matched")

    records, best = [], None
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        saved_val = checkpoint.get("val_videos")
        if saved_val is not None and sorted(saved_val) != sorted(val_videos):
            raise ValueError("Validation split mismatch: {}".format(checkpoint_path))
        mem_dim = checkpoint["student_state_dict"]["memory.memMatrix"].shape[0]
        decoder = OcbeAndDecoder(mem_dim=mem_dim, shrink_thres=0.0025).to(device)
        decoder.load_state_dict(checkpoint["student_state_dict"])
        metrics = validate_model(encoder, decoder, loader, dataset, device)
        record = {"checkpoint_path": os.path.abspath(checkpoint_path),
                  "epoch": int(checkpoint.get("epoch", -1)), **metrics}
        records.append(record)
        if is_better(record, best):
            best = record
        print(checkpoint_path, metrics)

    save_json(args.output, {
        "protocol": "training-only 80/20 video-level validation",
        "split_seed": args.split_seed,
        "train_videos": train_videos,
        "val_videos": val_videos,
        "test_labels_used": False,
        "scoring": {
            "cosine_map_weights": list(COSINE_MAP_WEIGHTS),
            "mse_map_weight": MSE_MAP_WEIGHT,
            "foreground_quantile": FOREGROUND_QUANTILE,
            "hard_ratio": HARD_RATIO,
            "background_mix": BACKGROUND_MIX,
            "background_keep": BACKGROUND_KEEP,
            "topk_ratio": TOPK_RATIO,
            "smooth_sigma": SMOOTH_SIGMA,
        },
        "checkpoints": records,
        "selected_checkpoint": best,
    })


def build_parser():
    parser = argparse.ArgumentParser(description="Leakage-free checkpoint validation")
    parser.add_argument("--dataset_type", choices=["ped2", "avenue", "shanghai"], required=True)
    parser.add_argument("--data_root", default=r"C:\dataset")
    parser.add_argument("--teacher_path", default="I3D_rgb_imagenet.pt")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", default="validation_results.json")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--val_pseudo_seed", type=int, default=2026)
    parser.add_argument("--val_stride", type=int, default=4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
