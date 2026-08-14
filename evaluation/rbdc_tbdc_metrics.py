"""Region- and track-based detection criteria for video anomaly maps.

The implementation follows the MERL EVAL protocol: 100 global thresholds,
8-connected detections, IoU >= 0.1 for a region hit, and at least 10% of a
track's annotated frames hit for a track detection.  False positives are
reported per test frame and the reported AUC is restricted to FPR <= 1.
"""

import csv
import math
import os
from collections import defaultdict

import numpy as np
from PIL import Image

try:
    from scipy import ndimage
except ImportError:  # Lightweight fallback used only by the standalone self-test.
    ndimage = None


DATASET_ANNOTATION_NAMES = {
    "ped2": "Ped2",
    "avenue": "Avenue",
    "shanghai": "ShanghaiTech",
}


def frame_scores_from_maps(score_maps, topk_ratio):
    """Aggregate final anomaly maps into frame scores without re-smoothing.

    The input maps must already contain every frozen spatial and temporal
    post-processing step.  Returning scores from these exact maps guarantees
    that frame AUC and RBDC/TBDC share one anomaly-map source.
    """
    ratio = float(topk_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("topk_ratio must be in (0, 1].")
    output = {}
    for video_name, maps in score_maps.items():
        current = np.asarray(maps)
        if current.ndim != 3:
            raise ValueError(
                "{} maps must have shape [T,H,W].".format(video_name)
            )
        if not bool(np.all(np.isfinite(current))):
            raise ValueError("{} maps contain non-finite values.".format(video_name))
        flattened = current.reshape(current.shape[0], -1)
        count = max(1, int(flattened.shape[1] * ratio))
        start = flattened.shape[1] - count
        topk = np.partition(flattened, start, axis=1)[:, start:]
        output[video_name] = np.asarray(topk.mean(axis=1), dtype=np.float32)
    return output


def _frame_key(path):
    """Return a stable frame key while accepting Windows backslashes."""
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    return os.path.splitext(name)[0]


def _annotation_file(annotation_root, dataset_type, video_name):
    dataset_dir = DATASET_ANNOTATION_NAMES[dataset_type]
    suffix = ".txt" if dataset_type == "shanghai" else "_gt.txt"
    return os.path.join(
        annotation_root, dataset_dir, "{}{}".format(video_name, suffix)
    )


def load_bbox_annotations(
    annotation_root,
    dataset_type,
    videos_meta,
    map_size,
):
    """Load dataset-specific track boxes and map them to anomaly-map coordinates.

    Ped2/Avenue use MERL's whitespace-separated
    ``frame track_id center_x center_y width height`` format. ShanghaiTech
    uses Georgescu et al.'s comma-separated
    ``track_id frame_idx x_min y_min x_max y_max`` format with zero-based
    frame indices. ``videos_meta`` is the ``dataset.videos`` mapping.
    """
    dataset_type = str(dataset_type).lower()
    if dataset_type not in DATASET_ANNOTATION_NAMES:
        raise ValueError("RBDC/TBDC supports Ped2, Avenue and ShanghaiTech.")
    if not os.path.isdir(annotation_root):
        raise FileNotFoundError(
            "Annotation root does not exist: {}".format(annotation_root)
        )
    if int(map_size) < 1:
        raise ValueError("map_size must be positive.")

    annotations = {}
    total_boxes = 0
    total_tracks = 0
    for video_name, meta in videos_meta.items():
        frames = list(meta["frame"])
        if not frames:
            raise RuntimeError("Video has no frames: {}".format(video_name))
        frame_lookup = {_frame_key(path): index for index, path in enumerate(frames)}
        numeric_lookup = {}
        for key, index in frame_lookup.items():
            try:
                numeric_lookup[int(key)] = index
            except ValueError:
                pass

        try:
            with Image.open(str(frames[0])) as first:
                original_width, original_height = first.size
        except (OSError, ValueError) as error:
            raise FileNotFoundError(
                "Unable to read frame: {}".format(frames[0])
            ) from error
        scale_x = float(map_size) / float(original_width)
        scale_y = float(map_size) / float(original_height)

        path = _annotation_file(annotation_root, dataset_type, video_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "Missing RBDC/TBDC annotation: {}".format(path)
            )
        per_frame = defaultdict(list)
        track_ids = set()
        with open(path, "r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                fields = (
                    [item.strip() for item in line.split(",")]
                    if dataset_type == "shanghai"
                    else line.split()
                )
                if len(fields) != 6:
                    raise ValueError(
                        "{}:{} must contain 6 fields, got {}.".format(
                            path, line_number, len(fields)
                        )
                    )
                if dataset_type == "shanghai":
                    try:
                        track_id = int(fields[0])
                        frame_index = int(fields[1])
                        left, top, right, bottom = map(float, fields[2:])
                    except ValueError as error:
                        raise ValueError(
                            "{}:{} contains invalid ShanghaiTech fields.".format(
                                path, line_number
                            )
                        ) from error
                    if not 0 <= frame_index < len(frames):
                        raise ValueError(
                            "{}:{} references zero-based frame {}, but {} has "
                            "{} frames.".format(
                                path,
                                line_number,
                                frame_index,
                                video_name,
                                len(frames),
                            )
                        )
                else:
                    frame_token, track_token = fields[:2]
                    key = _frame_key(frame_token)
                    frame_index = frame_lookup.get(key)
                    if frame_index is None:
                        try:
                            frame_index = numeric_lookup.get(int(key))
                        except ValueError:
                            frame_index = None
                    if frame_index is None:
                        raise ValueError(
                            "{}:{} references frame {!r}, which is not in {}.".format(
                                path, line_number, frame_token, video_name
                            )
                        )
                    try:
                        track_id = int(track_token)
                        center_x, center_y, width, height = map(float, fields[2:])
                    except ValueError as error:
                        raise ValueError(
                            "{}:{} contains invalid numeric fields.".format(
                                path, line_number
                            )
                        ) from error
                    left = center_x - width / 2.0
                    top = center_y - height / 2.0
                    right = center_x + width / 2.0
                    bottom = center_y + height / 2.0

                if track_id < 0:
                    raise ValueError(
                        "{}:{} has an invalid track ID.".format(
                            path, line_number
                        )
                    )
                if right <= left or bottom <= top:
                    raise ValueError(
                        "{}:{} contains an empty box.".format(
                            path, line_number
                        )
                    )

                per_frame[int(frame_index)].append(
                    {
                        "track_id": track_id,
                        "bbox": (
                            left * scale_x,
                            top * scale_y,
                            right * scale_x,
                            bottom * scale_y,
                        ),
                    }
                )
                track_ids.add(track_id)
                total_boxes += 1

        if not per_frame:
            raise ValueError("Annotation file contains no boxes: {}".format(path))
        if track_ids != set(range(max(track_ids) + 1)):
            raise ValueError(
                "{} uses non-contiguous track IDs; MERL EVAL requires IDs "
                "0..max_track_id.".format(path)
            )
        annotations[video_name] = {
            "frames": dict(per_frame),
            "track_ids": sorted(track_ids),
            "annotation_file": os.path.abspath(path),
            "original_height": int(original_height),
            "original_width": int(original_width),
            "frame_count": len(frames),
        }
        total_tracks += len(track_ids)

    return annotations, {
        "total_boxes": int(total_boxes),
        "total_tracks": int(total_tracks),
        "video_count": len(annotations),
        "bbox_conversion": (
            "aed_shanghai_zero_based_xyxy_then_axiswise_scale"
            if dataset_type == "shanghai"
            else "merl_continuous_center_size_then_axiswise_scale"
        ),
        "annotation_format": (
            "track_id,frame_idx,x_min,y_min,x_max,y_max"
            if dataset_type == "shanghai"
            else "frame_filename track_id center_x center_y width height"
        ),
    }


def _component_matches(score_map, threshold, boxes, iou_threshold):
    mask = np.asarray(score_map >= threshold, dtype=np.uint8)
    if ndimage is not None:
        labels, component_count = ndimage.label(
            mask, structure=np.ones((3, 3), dtype=np.uint8)
        )
    else:
        labels, component_count = _label_8_connected_fallback(mask)
    component_count = int(component_count)
    if component_count <= 0:
        return [], set(), 0

    count = component_count + 1
    component_areas = np.bincount(
        labels.reshape(-1), minlength=count
    )[1:].astype(np.float64)
    matched_components = set()
    matched_box_indices = []
    map_height, map_width = score_map.shape
    for box_index, item in enumerate(boxes):
        left, top, right, bottom = item["bbox"]
        x0 = max(0, min(map_width - 1, int(math.floor(left))))
        x1 = max(0, min(map_width - 1, int(math.floor(right))))
        y0 = max(0, min(map_height - 1, int(math.floor(top))))
        y1 = max(0, min(map_height - 1, int(math.floor(bottom))))
        if x1 < x0 or y1 < y0:
            continue
        intersection = np.bincount(
            labels[y0 : y1 + 1, x0 : x1 + 1].reshape(-1),
            minlength=count,
        )[1:].astype(np.float64)
        gt_area = max(1e-12, float((right - left) * (bottom - top)))
        union = component_areas + gt_area - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        hits = np.flatnonzero(iou >= float(iou_threshold))
        if hits.size:
            matched_box_indices.append(box_index)
            matched_components.update((hits + 1).tolist())
    false_positives = component_count - len(matched_components)
    return matched_box_indices, matched_components, int(false_positives)


def _label_8_connected_fallback(mask):
    """Small dependency-free 8-connected labeler for standalone self-tests."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    component = 0
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or labels[row, column] != 0:
                continue
            component += 1
            labels[row, column] = component
            stack = [(row, column)]
            while stack:
                current_row, current_column = stack.pop()
                for delta_row in (-1, 0, 1):
                    for delta_column in (-1, 0, 1):
                        if delta_row == 0 and delta_column == 0:
                            continue
                        next_row = current_row + delta_row
                        next_column = current_column + delta_column
                        if (
                            0 <= next_row < height
                            and 0 <= next_column < width
                            and mask[next_row, next_column]
                            and labels[next_row, next_column] == 0
                        ):
                            labels[next_row, next_column] = component
                            stack.append((next_row, next_column))
    return labels, component


def _auc_until_one(fpr, tpr):
    """Reproduce MERL compute_AUC.py on the official [0, 1] FPR range."""
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    if fpr.shape != tpr.shape or fpr.ndim != 1:
        raise ValueError("FPR and TPR must be equal-length one-dimensional arrays.")
    if not np.all(np.isfinite(fpr)) or not np.all(np.isfinite(tpr)):
        raise ValueError("FPR/TPR contain non-finite values.")

    # Curves are ordered from the lowest to highest threshold.  Match the
    # reference script by retaining the last point at/above FPR=1 and
    # linearly interpolating the crossing with the following point.
    crossing_index = -1
    for index, value in enumerate(fpr):
        if value >= 1.0:
            crossing_index = index
    if crossing_index == -1:
        fpr = np.insert(fpr, 0, 1.0)
        tpr = np.insert(tpr, 0, 1.0)
        crossing_index = 0
    if crossing_index + 1 >= len(fpr):
        fpr = np.append(fpr, 0.0)
        tpr = np.append(tpr, 0.0)

    x0, x1 = fpr[crossing_index], fpr[crossing_index + 1]
    y0, y1 = tpr[crossing_index], tpr[crossing_index + 1]
    if abs(float(x0 - x1)) <= 1e-15:
        interpolated_tpr = float(y0)
    else:
        interpolated_tpr = float(
            y1 + (y0 - y1) * (1.0 - x1) / (x0 - x1)
        )
    fpr = fpr[crossing_index:].copy()
    tpr = tpr[crossing_index:].copy()
    fpr[0] = 1.0
    tpr[0] = interpolated_tpr

    # Connected components may split/merge as the threshold changes.  MERL's
    # script deletes every next point that fails to strictly decrease FPR.
    bad_indices = [
        index + 1
        for index in range(len(fpr) - 1)
        if fpr[index + 1] >= fpr[index]
    ]
    if bad_indices:
        fpr = np.delete(fpr, bad_indices)
        tpr = np.delete(tpr, bad_indices)
    order = np.argsort(fpr)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(tpr[order], fpr[order]))
    return float(np.trapz(tpr[order], fpr[order]))


def evaluate_rbdc_tbdc(
    score_maps,
    annotations,
    threshold_count=100,
    iou_threshold=0.1,
    track_fraction=0.1,
    progress_callback=None,
):
    """Evaluate dictionaries of ``video -> [T,H,W]`` anomaly maps."""
    if int(threshold_count) < 2:
        raise ValueError("threshold_count must be at least 2.")
    if not 0.0 < float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1].")
    if not 0.0 < float(track_fraction) <= 1.0:
        raise ValueError("track_fraction must be in (0, 1].")
    if set(score_maps) != set(annotations):
        raise ValueError(
            "Map/annotation videos differ: maps={}, annotations={}.".format(
                sorted(score_maps), sorted(annotations)
            )
        )

    minimum = min(float(np.min(value)) for value in score_maps.values())
    maximum = max(float(np.max(value)) for value in score_maps.values())
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("Anomaly maps contain non-finite values.")
    if maximum <= minimum:
        maximum = minimum + 1e-12
    thresholds = np.linspace(minimum, maximum, int(threshold_count))

    total_frames = 0
    total_regions = 0
    track_lengths = {}
    for video_name, maps in score_maps.items():
        if maps.ndim != 3:
            raise ValueError("{} maps must have shape [T,H,W].".format(video_name))
        expected = int(annotations[video_name]["frame_count"])
        if int(maps.shape[0]) != expected:
            raise ValueError(
                "{} has {} maps but {} frames.".format(
                    video_name, maps.shape[0], expected
                )
            )
        total_frames += expected
        for frame_boxes in annotations[video_name]["frames"].values():
            total_regions += len(frame_boxes)
            for item in frame_boxes:
                key = (video_name, int(item["track_id"]))
                track_lengths[key] = track_lengths.get(key, 0) + 1
    if total_regions == 0 or not track_lengths:
        raise ValueError("No annotated regions/tracks were loaded.")

    region_tpr = []
    track_tpr = []
    false_positive_rate = []
    for threshold_index, threshold in enumerate(thresholds):
        detected_regions = 0
        false_positives = 0
        track_frame_hits = defaultdict(int)
        for video_name, maps in score_maps.items():
            frame_annotations = annotations[video_name]["frames"]
            for frame_index in range(maps.shape[0]):
                boxes = frame_annotations.get(frame_index, [])
                matched_box_indices, _, frame_fp = _component_matches(
                    maps[frame_index], threshold, boxes, iou_threshold
                )
                detected_regions += len(matched_box_indices)
                false_positives += frame_fp
                matched_tracks = {
                    (video_name, int(boxes[index]["track_id"]))
                    for index in matched_box_indices
                }
                for track_key in matched_tracks:
                    track_frame_hits[track_key] += 1

        detected_tracks = sum(
            1
            for track_key, length in track_lengths.items()
            if track_frame_hits.get(track_key, 0)
            >= float(track_fraction) * float(length)
        )
        region_tpr.append(float(detected_regions) / float(total_regions))
        track_tpr.append(float(detected_tracks) / float(len(track_lengths)))
        false_positive_rate.append(float(false_positives) / float(total_frames))
        if progress_callback is not None:
            progress_callback(threshold_index + 1, len(thresholds))

    return {
        "thresholds": thresholds.tolist(),
        "false_positives_per_frame": false_positive_rate,
        "region_tpr": region_tpr,
        "track_tpr": track_tpr,
        "rbdc_auc": _auc_until_one(false_positive_rate, region_tpr),
        "tbdc_auc": _auc_until_one(false_positive_rate, track_tpr),
        "total_test_frames": int(total_frames),
        "total_region_instances": int(total_regions),
        "total_tracks": int(len(track_lengths)),
        "threshold_count": int(threshold_count),
        "iou_threshold": float(iou_threshold),
        "track_fraction": float(track_fraction),
        "connectivity": 8,
        "auc_fpr_limit": 1.0,
    }


def save_curve_csv(path, metrics):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threshold",
                "false_positives_per_frame",
                "region_tpr",
                "track_tpr",
            ],
        )
        writer.writeheader()
        for values in zip(
            metrics["thresholds"],
            metrics["false_positives_per_frame"],
            metrics["region_tpr"],
            metrics["track_tpr"],
        ):
            writer.writerow(dict(zip(writer.fieldnames, values)))
