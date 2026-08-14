"""Test-only anomaly-map extraction and RBDC/TBDC evaluation."""

import argparse
import csv
import hashlib
import json
import os

import numpy as np
import torch
import torch.utils.data as data
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from tqdm import tqdm

from data import Reconstruction3DDataLoader
from rbdc_tbdc_metrics import (
    evaluate_rbdc_tbdc,
    frame_scores_from_maps,
    load_bbox_annotations,
    save_curve_csv,
)
from scoring import (
    build_foreground_mask,
    calculate_score_map,
    scoring_config_from_args,
    stable_config_hash,
)
from test import (
    align_student_features,
    checkpoint_identity,
    load_frame_labels,
    load_student,
    load_teacher,
    validate_label_lengths,
    validate_shanghai_masks,
)
from utils import (
    get_dataset_paths,
    restore_teacher_state_from_checkpoint,
    save_json,
    setup_seed,
)


FROZEN_SCORING_CONFIG_HASH = "a37f49c1637a"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_test_data(args):
    paths = get_dataset_paths(args.data_root, args.dataset_type)
    if not os.path.isdir(paths["test"]):
        raise FileNotFoundError("Test folder not found: {}".format(paths["test"]))
    dataset = Reconstruction3DDataLoader(
        paths["test"],
        transforms.Compose([transforms.ToTensor()]),
        resize_height=args.resize,
        resize_width=args.resize,
        num_frames=args.num_frames,
        dataset=args.dataset_type,
        img_extension=paths["extension"],
        train=False,
        jump=[1],
    )
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


def foreground_weighted_localization_maps(score_maps, images, config):
    """Return one spatial map whose top-k equals the configured frame score.

    For a non-zero hard ratio, the scalar scorer mixes a foreground-only
    hard score with a soft foreground-weighted score.  We use the same mixture
    to form a spatial map, then apply one positive scalar normalization per
    frame so its global top-k is exactly the configured mixed scalar score.
    """
    if not bool(config["use_foreground"]):
        return score_maps
    output = []
    topk_ratio = float(config["topk_ratio"])
    hard_ratio = float(config["hard_ratio"])

    def topk_mean(values):
        flattened = values.flatten()
        count = max(1, int(flattened.numel() * topk_ratio))
        return torch.topk(flattened, count, dim=0).values.mean()

    for batch_index in range(int(score_maps.shape[0])):
        current = score_maps[batch_index]
        mask = build_foreground_mask(
            images[batch_index : batch_index + 1],
            out_hw=int(current.shape[-1]),
            quantile=config["foreground_quantile"],
            min_area_ratio=config["min_area_ratio"],
            background_mix=config["background_mix"],
        )
        soft_map = current * (
            float(config["background_keep"])
            + (1.0 - float(config["background_keep"])) * mask
        )
        frame_maps = []
        for frame_index in range(int(current.shape[0])):
            foreground = mask[frame_index] > 0.5
            if not bool(foreground.any()):
                # Same zero-motion fallback as aggregate_clip_scores.
                frame_maps.append(current[frame_index])
                continue
            hard_score = topk_mean(current[frame_index][foreground])
            soft_score = topk_mean(soft_map[frame_index])
            target_score = (
                hard_ratio * hard_score + (1.0 - hard_ratio) * soft_score
            )
            hard_map = current[frame_index] * mask[frame_index]
            mixed_map = (
                hard_ratio * hard_map
                + (1.0 - hard_ratio) * soft_map[frame_index]
            )
            mixed_score = topk_mean(mixed_map)
            scale = torch.where(
                mixed_score.abs() > 1e-12,
                target_score / mixed_score.clamp_min(1e-12),
                torch.ones_like(mixed_score),
            )
            mixed_map = mixed_map * scale
            frame_maps.append(mixed_map)
        output.append(torch.stack(frame_maps, dim=0))
    return torch.stack(output, dim=0)


def sample_metadata(dataset, sample_cursor, batch_size):
    metadata = []
    for batch_index in range(int(batch_size)):
        sample_index = int(sample_cursor) + batch_index
        # Keep the user-requested Windows backslash path parsing.
        sample_path = str(dataset.samples[sample_index]).replace("/", "\\")
        metadata.append(
            (
                sample_path.split("\\")[-2],
                int(dataset.sample_start_indices[sample_index]),
            )
        )
    return metadata


def extract_maps(args, dataset, loader, scoring_config, map_directory):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required to extract RBDC/TBDC anomaly maps.")

    encoder = load_teacher(args.teacher_path, device)
    decoder, checkpoint = load_student(args.checkpoint, device)
    teacher_restore = restore_teacher_state_from_checkpoint(encoder, checkpoint)
    encoder.eval()
    decoder.eval()

    video_names = list(dataset.videos.keys())
    score_maps = {
        name: np.zeros(
            (
                int(dataset.videos[name]["length"]),
                int(scoring_config["map_size"]),
                int(scoring_config["map_size"]),
            ),
            dtype=np.float32,
        )
        for name in video_names
    }
    best_frame_scores = {
        name: np.full(
            int(dataset.videos[name]["length"]),
            -np.inf,
            dtype=np.float32,
        )
        for name in video_names
    }
    temporal_length = int(dataset._num_frames)
    sample_cursor = 0
    with torch.no_grad():
        for images in tqdm(loader, desc="RBDC/TBDC map extraction", ascii=True):
            images = images.to(device, non_blocking=True)
            teacher_features = encoder(images)
            student_raw = decoder(*teacher_features)
            student_features = align_student_features(
                teacher_features, student_raw[:3]
            )
            batch_maps = calculate_score_map(
                teacher_features,
                student_features,
                temporal_length=temporal_length,
                output_size=scoring_config["map_size"],
                cosine_map_weights=scoring_config["cosine_map_weights"],
                mse_map_weight=scoring_config["mse_map_weight"],
                spatial_smooth_kernel=scoring_config["spatial_smooth_kernel"],
            )
            batch_maps = foreground_weighted_localization_maps(
                batch_maps, images, scoring_config
            )
            flattened = batch_maps.flatten(2)
            topk_count = max(
                1,
                int(
                    flattened.shape[-1]
                    * float(scoring_config["topk_ratio"])
                ),
            )
            batch_frame_scores = torch.topk(
                flattened, topk_count, dim=-1
            ).values.mean(dim=-1)
            batch_maps = batch_maps.cpu().numpy()
            batch_frame_scores = batch_frame_scores.cpu().numpy()
            metadata = sample_metadata(dataset, sample_cursor, images.shape[0])
            for batch_index, (video_name, start_frame) in enumerate(metadata):
                video_length = score_maps[video_name].shape[0]
                end_frame = min(start_frame + temporal_length, video_length)
                if start_frame < end_frame:
                    current = batch_maps[batch_index, : end_frame - start_frame]
                    current_scores = batch_frame_scores[
                        batch_index, : end_frame - start_frame
                    ]
                    destination_scores = best_frame_scores[video_name][
                        start_frame:end_frame
                    ]
                    replace = current_scores > destination_scores
                    if bool(np.any(replace)):
                        destination_maps = score_maps[video_name][
                            start_frame:end_frame
                        ]
                        destination_maps[replace] = current[replace]
                        destination_scores[replace] = current_scores[replace]
            sample_cursor += int(images.shape[0])

    if sample_cursor != len(dataset.samples):
        raise RuntimeError(
            "Sample accounting mismatch: processed {}, expected {}.".format(
                sample_cursor, len(dataset.samples)
            )
        )

    os.makedirs(map_directory, exist_ok=True)
    sigma = float(scoring_config["smooth_sigma"])
    for video_name in video_names:
        current = score_maps[video_name]
        raw_scores = best_frame_scores[video_name]
        if not bool(np.all(np.isfinite(raw_scores))):
            missing = int(np.sum(~np.isfinite(raw_scores)))
            raise RuntimeError(
                "{} frames in {} were not covered by any clip.".format(
                    missing, video_name
                )
            )
        if sigma > 0.0:
            smoothed_scores = gaussian_filter(raw_scores, sigma=sigma)
            scale = np.divide(
                smoothed_scores,
                raw_scores,
                out=np.ones_like(smoothed_scores),
                where=np.abs(raw_scores) > 1e-12,
            )
            # Preserve the selected clip's localization pattern while making
            # the map's frozen top-k aggregation exactly follow the scalar
            # Gaussian smoothing used by test.py.
            current = current * scale[:, None, None]
        current = np.asarray(current, dtype=np.float32)
        np.save(os.path.join(map_directory, video_name + ".npy"), current)
        score_maps[video_name] = current
    return score_maps, checkpoint, teacher_restore


def load_cached_maps(map_directory, dataset):
    maps = {}
    map_size = None
    for video_name, meta in dataset.videos.items():
        path = os.path.join(map_directory, video_name + ".npy")
        if not os.path.isfile(path):
            raise FileNotFoundError("Cached map not found: {}".format(path))
        current = np.load(path, mmap_mode="r")
        if current.ndim != 3 or int(current.shape[0]) != int(meta["length"]):
            raise ValueError(
                "Cached map shape mismatch for {}: {}.".format(
                    video_name, current.shape
                )
            )
        if map_size is None:
            map_size = tuple(current.shape[1:])
        if tuple(current.shape[1:]) != map_size:
            raise ValueError("Cached maps do not share one spatial shape.")
        maps[video_name] = current
    return maps


def evaluate_frame_auc(frame_scores, labels_dict, video_names):
    """Calculate Micro/Macro frame AUC from the final anomaly maps."""
    all_scores = []
    all_labels = []
    per_video_auc = {}
    for video_name in video_names:
        scores = np.asarray(frame_scores[video_name]).reshape(-1)
        labels = (
            np.asarray(labels_dict[video_name]).reshape(-1) > 0
        ).astype(np.uint8)
        if scores.size != labels.size:
            raise ValueError(
                "Frame score/label mismatch for {}: {} vs {}.".format(
                    video_name, scores.size, labels.size
                )
            )
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
        if np.unique(labels).size == 2:
            per_video_auc[video_name] = float(
                roc_auc_score(labels, scores)
            )
    if np.unique(np.asarray(all_labels)).size != 2:
        raise ValueError("The complete frame-label set must contain two classes.")
    return {
        "micro_frame_auc": float(roc_auc_score(all_labels, all_scores)),
        "macro_frame_auc": (
            float(np.mean(list(per_video_auc.values())))
            if per_video_auc
            else None
        ),
        "macro_valid_video_count": int(len(per_video_auc)),
        "test_video_count": int(len(video_names)),
        "test_frame_count": int(len(all_labels)),
        "per_video_auc": per_video_auc,
    }


def validate_formal_protocol(args, scoring_hash):
    """Reject any formal run that drifts from the pre-registered protocol."""
    if not bool(args.formal_protocol):
        return None
    if scoring_hash != FROZEN_SCORING_CONFIG_HASH:
        raise ValueError(
            "Formal RBDC/TBDC evaluation requires frozen scoring hash {}, got {}.".format(
                FROZEN_SCORING_CONFIG_HASH, scoring_hash
            )
        )
    fixed_protocol = {
        "threshold_count": 100,
        "iou_threshold": 0.1,
        "track_fraction": 0.1,
    }
    for key, expected in fixed_protocol.items():
        actual = getattr(args, key)
        if abs(float(actual) - float(expected)) > 1e-12:
            raise ValueError(
                "Formal RBDC/TBDC evaluation fixes {}={}, got {}.".format(
                    key, expected, actual
                )
            )
    summary_path = str(args.checkpoint_training_summary).strip()
    if not summary_path or not os.path.isfile(summary_path):
        raise FileNotFoundError(
            "Formal RBDC/TBDC evaluation requires checkpoint training_summary.json."
        )
    if os.path.dirname(os.path.abspath(summary_path)) != os.path.dirname(
        os.path.abspath(args.checkpoint)
    ):
        raise ValueError(
            "Checkpoint and training summary must be in the same run directory."
        )
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("completed") is not True:
        raise ValueError("Checkpoint training summary is not completed.")
    if summary.get("selection_uses_test_labels") is not False:
        raise ValueError(
            "Formal checkpoint selection must not use test labels."
        )
    if str(summary.get("scoring_config_hash", "")) != scoring_hash:
        raise ValueError(
            "Training summary scoring hash does not match the formal run."
        )
    return {
        "training_summary": os.path.abspath(summary_path),
        "training_summary_sha256": file_sha256(summary_path),
        "completed": True,
        "selection_uses_test_labels": False,
        "scoring_config_hash": scoring_hash,
    }


def write_summary_csv(path, metrics):
    row = {
        "dataset": metrics["dataset"],
        "micro_frame_auc": metrics["micro_frame_auc"],
        "micro_frame_auc_percent": metrics["micro_frame_auc_percent"],
        "macro_frame_auc": metrics["macro_frame_auc"],
        "macro_frame_auc_percent": metrics["macro_frame_auc_percent"],
        "macro_valid_video_count": metrics["macro_valid_video_count"],
        "rbdc_auc": metrics["rbdc_auc"],
        "rbdc_auc_percent": metrics["rbdc_auc_percent"],
        "tbdc_auc": metrics["tbdc_auc"],
        "tbdc_auc_percent": metrics["tbdc_auc_percent"],
        "total_test_frames": metrics["total_test_frames"],
        "total_region_instances": metrics["total_region_instances"],
        "total_tracks": metrics["total_tracks"],
        "iou_threshold": metrics["iou_threshold"],
        "track_fraction": metrics["track_fraction"],
        "threshold_count": metrics["threshold_count"],
        "checkpoint_hash": metrics["checkpoint_identifier"]["file_sha256"][:10],
        "checkpoint_identity_hash": metrics["checkpoint_identifier"]["hash"],
        "scoring_config_hash": metrics["scoring_config_hash"],
        "localization_config_hash": metrics["localization_config_hash"],
        "maps_reused": metrics["maps_reused"],
    }
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def evaluate(args):
    setup_seed(args.seed)
    if args.dataset_type not in {"ped2", "avenue", "shanghai"}:
        raise ValueError("RBDC/TBDC evaluation supports ped2, avenue and shanghai.")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid batch_size or num_workers.")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError("Checkpoint not found: {}".format(args.checkpoint))

    scoring_config = scoring_config_from_args(args)
    scoring_hash = stable_config_hash(scoring_config)
    expected_scoring_hash = str(args.expected_scoring_hash).strip()
    if not expected_scoring_hash:
        expected_scoring_hash = FROZEN_SCORING_CONFIG_HASH
    if scoring_hash != expected_scoring_hash:
        raise ValueError(
            "Scoring configuration drift: got {}, expected {}.".format(
                scoring_hash, expected_scoring_hash
            )
        )
    formal_audit = validate_formal_protocol(args, scoring_hash)
    dataset, loader = build_test_data(args)
    dataset_paths = get_dataset_paths(args.data_root, args.dataset_type)
    video_names = list(dataset.videos.keys())
    labels_dict, labels_source = load_frame_labels(
        dataset_type=args.dataset_type,
        labels_path=args.labels_path,
        dataset_paths=dataset_paths,
        video_names=video_names,
        videos_meta=dataset.videos,
    )
    validate_label_lengths(labels_dict, video_names, dataset.videos)
    checked_mask_frames = 0
    if args.dataset_type == "shanghai":
        checked_mask_frames = validate_shanghai_masks(
            dataset_paths, video_names
        )
    annotations, annotation_audit = load_bbox_annotations(
        args.annotations_root,
        args.dataset_type,
        dataset.videos,
        scoring_config["map_size"],
    )

    checkpoint_cpu = torch.load(args.checkpoint, map_location="cpu")
    if bool(args.formal_protocol) and int(
        checkpoint_cpu.get("best_epoch", -1)
    ) < 0:
        raise ValueError(
            "Formal RBDC/TBDC evaluation requires a checkpoint with an auditable best_epoch."
        )
    checkpoint_id = checkpoint_identity(args.checkpoint, checkpoint_cpu)
    checkpoint_digest = file_sha256(args.checkpoint)
    checkpoint_id = dict(checkpoint_id)
    checkpoint_id["file_sha256"] = checkpoint_digest
    localization_config = {
        "dataset": args.dataset_type,
        "checkpoint_identifier": checkpoint_id,
        "resize": int(args.resize),
        "num_frames": int(args.num_frames),
        "scoring": scoring_config,
        "overlap_reduction": "map_from_clip_with_max_frozen_frame_score",
        "foreground_map": (
            "hard_soft_spatial_mix_normalized_to_configured_frame_score_"
            "with_empty_mask_global_fallback"
        ),
        "temporal_smoothing": (
            "per_frame_map_rescaling_to_frozen_gaussian_smoothed_topk_score"
        ),
        "temporal_smoothing_sigma": float(scoring_config["smooth_sigma"]),
        "bbox_scaling": "separate_original_width_and_height_to_map_size",
    }
    localization_hash = stable_config_hash(localization_config)
    output_dir = os.path.abspath(args.output_dir)
    map_directory = os.path.join(output_dir, "maps_" + localization_hash)
    manifest_path = os.path.join(map_directory, "manifest.json")

    maps_reused = False
    teacher_restore = {
        "teacher_finetune_mode": "from_cache",
        "teacher_state_restored": None,
    }
    if bool(args.reuse_maps) and os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("localization_config_hash") != localization_hash:
            raise ValueError("Cached RBDC/TBDC map manifest has a different config.")
        score_maps = load_cached_maps(map_directory, dataset)
        maps_reused = True
        print("Reusing cached anomaly maps:", map_directory)
    else:
        score_maps, extracted_checkpoint, teacher_restore = extract_maps(
            args, dataset, loader, scoring_config, map_directory
        )
        extracted_id = checkpoint_identity(args.checkpoint, extracted_checkpoint)
        extracted_id = dict(extracted_id)
        extracted_id["file_sha256"] = checkpoint_digest
        if extracted_id != checkpoint_id:
            raise RuntimeError("Checkpoint identity changed during extraction.")
        save_json(
            manifest_path,
            {
                "completed": True,
                "dataset": args.dataset_type,
                "video_names": list(dataset.videos.keys()),
                "localization_config": localization_config,
                "localization_config_hash": localization_hash,
                "formal_protocol": bool(args.formal_protocol),
                "checkpoint_selection_audit": formal_audit,
                "shared_anomaly_maps_for_all_metrics": True,
                "test_labels_used_for_model_selection": False,
                "test_annotations_used_for_evaluation_only": True,
            },
        )

    frame_scores = frame_scores_from_maps(
        score_maps, scoring_config["topk_ratio"]
    )
    frame_metrics = evaluate_frame_auc(
        frame_scores, labels_dict, video_names
    )

    progress = tqdm(total=args.threshold_count, desc="RBDC/TBDC thresholds", ascii=True)

    def update_progress(completed, total):
        del total
        progress.update(max(0, int(completed) - int(progress.n)))

    protocol = evaluate_rbdc_tbdc(
        score_maps,
        annotations,
        threshold_count=args.threshold_count,
        iou_threshold=args.iou_threshold,
        track_fraction=args.track_fraction,
        progress_callback=update_progress,
    )
    progress.close()

    metrics = dict(protocol)
    metrics.update(
        {
            "dataset": args.dataset_type,
            "micro_frame_auc": frame_metrics["micro_frame_auc"],
            "micro_frame_auc_percent": 100.0
            * float(frame_metrics["micro_frame_auc"]),
            "macro_frame_auc": frame_metrics["macro_frame_auc"],
            "macro_frame_auc_percent": (
                100.0 * float(frame_metrics["macro_frame_auc"])
                if frame_metrics["macro_frame_auc"] is not None
                else None
            ),
            "macro_valid_video_count": frame_metrics[
                "macro_valid_video_count"
            ],
            "test_video_count": frame_metrics["test_video_count"],
            "test_frame_count": frame_metrics["test_frame_count"],
            "per_video_auc": frame_metrics["per_video_auc"],
            "rbdc_auc_percent": 100.0 * float(protocol["rbdc_auc"]),
            "tbdc_auc_percent": 100.0 * float(protocol["tbdc_auc"]),
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_identifier": checkpoint_id,
            "checkpoint_epoch": int(checkpoint_cpu.get("epoch", -1)),
            "checkpoint_best_epoch": int(checkpoint_cpu.get("best_epoch", -1)),
            "scoring": scoring_config,
            "scoring_config_hash": scoring_hash,
            "localization_config": localization_config,
            "localization_config_hash": localization_hash,
            "annotation_root": os.path.abspath(args.annotations_root),
            "annotation_audit": annotation_audit,
            "frame_labels_source": labels_source,
            "shanghai_mask_consistency_checked_frames": int(
                checked_mask_frames
            ),
            "maps_reused": maps_reused,
            "map_directory": map_directory,
            "teacher_restore": teacher_restore,
            "training_invoked": False,
            "dataloader_shuffle": False,
            "formal_protocol": bool(args.formal_protocol),
            "formal_protocol_lock": (
                {
                    "scoring_config_hash": FROZEN_SCORING_CONFIG_HASH,
                    "threshold_count": 100,
                    "iou_alpha": 0.1,
                    "track_beta": 0.1,
                    "false_positives_per_frame_range": [0.0, 1.0],
                    "connectivity": 8,
                    "threshold_selection": "automatic_global_scan",
                }
                if bool(args.formal_protocol)
                else None
            ),
            "checkpoint_selection_audit": formal_audit,
            "selection_scope": (
                "formal_validation_selected_frozen"
                if bool(args.formal_protocol)
                else "exploratory_test_evaluation"
            ),
            "shared_anomaly_maps_for_all_metrics": True,
            "frame_score_source": (
                "topk_mean_from_final_localization_maps_after_frozen_"
                "temporal_smoothing"
            ),
            "additional_frame_score_smoothing": False,
            "test_annotations_used_for_evaluation_only": True,
            "test_frame_labels_used_for_evaluation_only": True,
            "protocol_reference": "MERL EVAL region/track criteria",
        }
    )
    os.makedirs(output_dir, exist_ok=True)
    prefix_base = (
        "rbdc_tbdc_formal_four_metrics"
        if bool(args.formal_protocol)
        else "rbdc_tbdc"
    )
    prefix = "{}_{}_ckpt{}_cfg{}".format(
        prefix_base,
        args.dataset_type, checkpoint_digest[:10], scoring_hash
    )
    metrics_path = os.path.join(output_dir, prefix + "_metrics.json")
    curve_path = os.path.join(output_dir, prefix + "_curve.csv")
    scores_path = os.path.join(output_dir, prefix + "_frame_scores.npz")
    summary_path = os.path.join(
        output_dir, "{}_{}.csv".format(prefix_base, args.dataset_type)
    )
    existing = [
        path
        for path in (metrics_path, curve_path, scores_path, summary_path)
        if os.path.exists(path)
    ]
    if existing and not bool(args.overwrite):
        raise FileExistsError(
            "RBDC/TBDC result already exists. Use --overwrite 1 to replace only "
            "the reports; cached maps remain reusable: {}".format(existing)
        )
    save_json(metrics_path, metrics)
    save_curve_csv(curve_path, metrics)
    np.savez_compressed(
        scores_path,
        video_names=np.asarray(video_names, dtype=object),
        scores=np.asarray(
            [frame_scores[name] for name in video_names], dtype=object
        ),
        labels=np.asarray(
            [
                np.asarray(labels_dict[name]).reshape(-1)
                for name in video_names
            ],
            dtype=object,
        ),
        scoring_config_hash=np.asarray(scoring_hash),
        localization_config_hash=np.asarray(localization_hash),
        checkpoint_sha256=np.asarray(checkpoint_digest),
    )
    write_summary_csv(summary_path, metrics)
    print(json.dumps({
        "dataset": args.dataset_type,
        "Micro_AUC_percent": metrics["micro_frame_auc_percent"],
        "Macro_AUC_percent": metrics["macro_frame_auc_percent"],
        "RBDC_AUC_percent": metrics["rbdc_auc_percent"],
        "TBDC_AUC_percent": metrics["tbdc_auc_percent"],
        "regions": metrics["total_region_instances"],
        "tracks": metrics["total_tracks"],
    }, ensure_ascii=False, indent=2))
    print("Metrics:", metrics_path)
    print("Curve:", curve_path)
    print("Frame scores:", scores_path)
    print("Summary:", summary_path)
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(
        description="Test-only RBDC/TBDC evaluation."
    )
    parser.add_argument(
        "--dataset_type",
        choices=["ped2", "avenue", "shanghai"],
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=r"C:\dataset")
    parser.add_argument("--annotations_root", required=True)
    parser.add_argument("--teacher_path", default="I3D_rgb_imagenet.pt")
    parser.add_argument("--labels_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map_size", type=int, default=64)
    parser.add_argument("--cosine_map_weights", default="0.5,1.0,1.2")
    parser.add_argument("--mse_map_weight", type=float, default=1.0)
    parser.add_argument("--spatial_smooth_kernel", type=int, default=3)
    parser.add_argument("--topk_ratio", type=float, default=0.1)
    parser.add_argument("--smooth_sigma", type=float, default=9.0)
    parser.add_argument("--use_foreground", type=int, choices=[0, 1], default=1)
    parser.add_argument("--foreground_quantile", type=float, default=0.75)
    parser.add_argument("--min_area_ratio", type=float, default=0.01)
    parser.add_argument("--background_keep", type=float, default=0.2)
    parser.add_argument("--hard_ratio", type=float, default=0.6)
    parser.add_argument("--background_mix", type=float, default=0.7)
    parser.add_argument(
        "--expected_scoring_hash", default="1c42b3673178"
    )
    parser.add_argument("--threshold_count", type=int, default=100)
    parser.add_argument("--iou_threshold", type=float, default=0.1)
    parser.add_argument("--track_fraction", type=float, default=0.1)
    parser.add_argument("--reuse_maps", type=int, choices=[0, 1], default=1)
    parser.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--formal_protocol", type=int, choices=[0, 1], default=0
    )
    parser.add_argument("--checkpoint_training_summary", default="")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
