import torch
import torch.nn as nn
from data import *
import argparse
import os
from pathlib import Path
from loss import get_ano_map as get_ano_map_with_mse

import torchvision.transforms as transforms
from model import OcbeAndDecoder
from I3D_model import InceptionI3d
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F


def _build_fg_mask(images, out_hw=64, q=0.75, min_area_ratio=0.01, bg_mix=0.7):
    """
    images: (B, C, T, H, W)，当前输入已经是 [-1, 1]
    return: (T, out_hw, out_hw) 的 0/1 前景 mask
    """
    x = images[0]  # (C, T, H, W)

    if x.shape[0] >= 3:
        gray = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]  # (T, H, W)
    else:
        gray = x.mean(dim=0)

    gray = F.interpolate(
        gray.unsqueeze(1),
        size=(out_hw, out_hw),
        mode="bilinear",
        align_corners=True
    ).squeeze(1)  # (T, out_hw, out_hw)

    bg = gray.median(dim=0).values
    bg_diff = (gray - bg.unsqueeze(0)).abs()

    prev = torch.cat([gray[:1], gray[:-1]], dim=0)
    motion = (gray - prev).abs()

    fg_energy = bg_mix * bg_diff + (1.0 - bg_mix) * motion  # (T, out_hw, out_hw)

    thr = torch.quantile(
        fg_energy.flatten(1), q, dim=1, keepdim=True
    ).view(-1, 1, 1)

    fg_mask = (fg_energy >= thr).float()

    fg_mask = (
        F.avg_pool2d(fg_mask.unsqueeze(1), kernel_size=3, stride=1, padding=1) > 0
    ).float().squeeze(1)

    min_pixels = max(1, int(out_hw * out_hw * float(min_area_ratio)))
    cur_pixels = fg_mask.flatten(1).sum(dim=1)

    for t in range(fg_mask.shape[0]):
        if int(cur_pixels[t].item()) < min_pixels:
            flat = fg_energy[t].flatten()
            top_idx = torch.topk(flat, k=min_pixels, dim=0).indices
            tmp = torch.zeros_like(flat)
            tmp[top_idx] = 1.0
            fg_mask[t] = tmp.view(out_hw, out_hw)

    return fg_mask


def _build_labels_dict(labels_raw, video_names, videos_meta):
    """
    将 frame_labels_*.npy 统一成 {video_name: 1D ndarray} 的格式。
    兼容常见保存方式：dict / list-of-arrays / object ndarray / 2D ndarray / 1D concat。
    """
    labels = labels_raw

    if isinstance(labels, np.ndarray) and labels.shape == () and labels.dtype == object:
        labels = labels.item()

    if isinstance(labels, np.ndarray) and labels.dtype == object and labels.shape == (1,):
        labels = labels[0]

    if isinstance(labels, np.ndarray) and labels.ndim == 2 and 1 in labels.shape:
        labels = labels.reshape(-1)

    if isinstance(labels, dict):
        return {vn: np.asarray(labels[vn]).squeeze() for vn in video_names}

    if isinstance(labels, (list, tuple)):
        if len(labels) == len(video_names):
            return {vn: np.asarray(labels[i]).squeeze() for i, vn in enumerate(video_names)}

    if isinstance(labels, np.ndarray) and labels.dtype == object and labels.ndim == 1:
        if len(labels) == len(video_names):
            return {vn: np.asarray(labels[i]).squeeze() for i, vn in enumerate(video_names)}

    if isinstance(labels, np.ndarray) and labels.ndim == 2 and labels.shape[0] == len(video_names):
        return {vn: np.asarray(labels[i]).squeeze() for i, vn in enumerate(video_names)}

    if isinstance(labels, np.ndarray) and labels.ndim == 1:
        out = {}
        off = 0
        total_need = sum(int(videos_meta[vn]["length"]) for vn in video_names)
        if len(labels) < total_need:
            raise ValueError(f"Label length too short: got {len(labels)}, need {total_need}.")
        for vn in video_names:
            L = int(videos_meta[vn]["length"])
            out[vn] = np.asarray(labels[off: off + L]).squeeze()
            off += L
        return out

    raise ValueError(
        f"Unrecognized label format: type={type(labels)}, shape={getattr(labels, 'shape', None)}"
    )


def Test(
    ckp_dir,
    data_dir,
    resize,
    dataset_type,
    k=0.1,
    smooth_sigma=9.0,
    use_fg_mask=True,
    fg_q=0.75,
    min_area_ratio=0.01,
    bg_keep=0.20,
    hard_ratio=0.60,
    fg_alpha=0.70,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_extension = ".tif" if dataset_type == "ped2" else ".jpg"

    # 教师网络
    encoder = InceptionI3d(num_classes=400, in_channels=3, dropout_keep_prob=0.5)
    encoder.load_state_dict(torch.load("I3D_rgb_imagenet.pt", map_location=device))
    encoder.to(device).eval()

    # 学生网络
    ckpt = torch.load(ckp_dir, map_location=device)
    mem_dim = ckpt["student_state_dict"]["memory.memMatrix"].shape[0]
    ocbe_decoder = OcbeAndDecoder(in_channels_list=[480, 832, 1024], mem_dim=mem_dim).to(device)
    ocbe_decoder.load_state_dict(ckpt["student_state_dict"])
    ocbe_decoder.to(device).eval()

    test_dataset = Reconstruction3DDataLoader(
        data_dir,
        transforms.Compose([transforms.ToTensor()]),
        resize_height=resize,
        resize_width=resize,
        dataset=dataset_type,
        img_extension=img_extension,
        train=False, jump=[1],
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )

    labels_raw = np.load(f"./frame_labels_{dataset_type}.npy", allow_pickle=True)
    video_names = list(test_dataset.videos.keys())
    labels_dict = _build_labels_dict(labels_raw, video_names, test_dataset.videos)

    # 每帧分数容器，滑窗内取 max 聚合
    frame_scores = {
        vn: np.zeros(int(test_dataset.videos[vn]["length"]), dtype=np.float32)
        for vn in video_names
    }

    T = int(getattr(test_dataset, "_num_frames", 16))

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_loader, desc="Testing", leave=False, disable=True)):
            if isinstance(sample, np.ndarray):
                sample = torch.from_numpy(sample)
            images = sample.to(device)

            t_feat1, t_feat2, t_feat3 = encoder(images)
            s_feat1, s_feat2, s_feat3, _ = ocbe_decoder(t_feat1, t_feat2, t_feat3)

            c1, c2, c3 = t_feat1.shape[1], t_feat2.shape[1], t_feat3.shape[1]
            s_feat1 = s_feat1[:, :c1, ...]
            s_feat2 = s_feat2[:, :c2, ...]
            s_feat3 = s_feat3[:, :c3, ...]

            ano_map1, _, mse1 = get_ano_map_with_mse(t_feat1, s_feat1)
            ano_map2, _, mse2 = get_ano_map_with_mse(t_feat2, s_feat2)
            ano_map3, _, mse3 = get_ano_map_with_mse(t_feat3, s_feat3)

            ano_map1 = F.interpolate(ano_map1, size=(T, 64, 64), mode="trilinear", align_corners=True)
            ano_map2 = F.interpolate(ano_map2, size=(T, 64, 64), mode="trilinear", align_corners=True)
            ano_map3 = F.interpolate(ano_map3, size=(T, 64, 64), mode="trilinear", align_corners=True)
            mse1 = F.interpolate(mse1, size=(T, 64, 64), mode="trilinear", align_corners=True)
            mse2 = F.interpolate(mse2, size=(T, 64, 64), mode="trilinear", align_corners=True)
            mse3 = F.interpolate(mse3, size=(T, 64, 64), mode="trilinear", align_corners=True)

            m1 = ano_map1[0, 0]; e1 = mse1[0, 0]
            m2 = ano_map2[0, 0]; e2 = mse2[0, 0]
            m3 = ano_map3[0, 0]; e3 = mse3[0, 0]

            def smooth_hw(m):
                return F.avg_pool2d(m.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)

            m1 = smooth_hw(m1); e1 = smooth_hw(e1)
            m2 = smooth_hw(m2); e2 = smooth_hw(e2)
            m3 = smooth_hw(m3); e3 = smooth_hw(e3)

            score_map = 0.5 * m1 + 1.0 * m2 + 1.5 * m3 + 1.0 * torch.log1p(e3)  # (T, 64, 64)

            if use_fg_mask:
                fg_mask = _build_fg_mask(
                    images,
                    out_hw=64,
                    q=fg_q,
                    min_area_ratio=min_area_ratio,
                    bg_mix=fg_alpha,
                )

                scores_t = []
                for t_idx in range(T):
                    cur_mask = fg_mask[t_idx] > 0.5
                    hard_vals = score_map[t_idx][cur_mask]
                    if hard_vals.numel() == 0:
                        hard_vals = score_map[t_idx].flatten()
                    hard_topk_num = max(1, int(hard_vals.numel() * float(k)))
                    hard_score = torch.topk(hard_vals, hard_topk_num, dim=0).values.mean()

                    soft_map = score_map[t_idx] * (bg_keep + (1.0 - bg_keep) * fg_mask[t_idx])
                    soft_vals = soft_map.flatten()
                    soft_topk_num = max(1, int(soft_vals.numel() * float(k)))
                    soft_score = torch.topk(soft_vals, soft_topk_num, dim=0).values.mean()

                    cur_score = hard_ratio * hard_score + (1.0 - hard_ratio) * soft_score
                    scores_t.append(cur_score)

                scores_t = torch.stack(scores_t).detach().cpu().numpy()
            else:
                per_t = score_map.flatten(1)
                topk_num = max(1, int(per_t.shape[1] * float(k)))
                topk_vals, _ = torch.topk(per_t, topk_num, dim=1)
                scores_t = topk_vals.mean(dim=1).detach().cpu().numpy()

            # 路径解析：获取 video_name 和 start_frame
            sample_path = test_dataset.samples[idx]
            norm_path = str(sample_path).replace("\\", "/")
            parts = norm_path.split("/")
            video_name = parts[-2]

            stem = Path(parts[-1]).stem
            try:
                start_frame = int(stem) - 1
            except ValueError:
                continue
            if start_frame < 0:
                start_frame = 0

            # 滑窗回填：max 聚合
            L = frame_scores[video_name].shape[0]
            s = start_frame
            e = min(start_frame + T, L)
            if s < e:
                cur = scores_t[: (e - s)]
                frame_scores[video_name][s:e] = np.maximum(frame_scores[video_name][s:e], cur)

    # 计算 AUROC
    all_scores = []
    all_labels = []
    for vn in video_names:
        sc = frame_scores[vn]
        if smooth_sigma is not None and float(smooth_sigma) > 0:
            sc = gaussian_filter(sc, sigma=float(smooth_sigma))

        lb = np.asarray(labels_dict[vn]).squeeze()
        lb = (lb > 0).astype(np.int32)

        n = min(len(sc), len(lb))
        all_scores.extend(sc[:n].tolist())
        all_labels.extend(lb[:n].tolist())

    auroc_img = roc_auc_score(all_labels, all_scores)
    return auroc_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    test_folder = os.path.join("UCSDped2", "Test")
    parser.add_argument("--k", type=float, default=0.10)
    parser.add_argument("--sigma", type=float, default=9)
    parser.add_argument("--use_fg_mask", type=int, default=1)
    parser.add_argument("--fg_q", type=float, default=0.75)
    parser.add_argument("--min_area_ratio", type=float, default=0.01)
    parser.add_argument("--bg_keep", type=float, default=0.20)
    parser.add_argument("--hard_ratio", type=float, default=0.60)
    parser.add_argument("--fg_alpha", type=float, default=0.70)

    args = parser.parse_args()

    auroc_img = Test(
        ckp_dir="./checkpoints/I3Dtrain_num0_lr0.002_bs4/epoch8.pth",
        data_dir=test_folder,
        resize=256,
        dataset_type="ped2",
        k=args.k,
        smooth_sigma=args.sigma if args.sigma > 0 else 0.0,
        use_fg_mask=bool(args.use_fg_mask),
        fg_q=args.fg_q,
        min_area_ratio=args.min_area_ratio,
        bg_keep=args.bg_keep,
        hard_ratio=args.hard_ratio,
        fg_alpha=args.fg_alpha,
    )
    print(auroc_img)
