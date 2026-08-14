"""Collect public training summaries into one CSV file."""

import argparse
import csv
import json
import os


FIELDS = [
    "run_name", "protocol", "epochs", "train_video_count",
    "val_video_count", "best_epoch", "best_validation_auc",
    "best_separation", "best_normal_objective", "test_labels_used",
    "summary_path",
]


def collect(checkpoint_root):
    records = []
    for root, _, files in os.walk(checkpoint_root):
        if "training_summary.json" not in files:
            continue
        path = os.path.join(root, "training_summary.json")
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        best = summary.get("best_validation") or {}
        records.append({
            "run_name": os.path.basename(root),
            "protocol": summary.get("protocol", ""),
            "epochs": summary.get("epochs", ""),
            "train_video_count": len(summary.get("train_videos", [])),
            "val_video_count": len(summary.get("val_videos", [])),
            "best_epoch": best.get("epoch", ""),
            "best_validation_auc": best.get("aligned_scoring_auc", ""),
            "best_separation": best.get("aligned_effect_size", ""),
            "best_normal_objective": best.get("normal_objective", ""),
            "test_labels_used": summary.get("test_labels_used", False),
            "summary_path": os.path.abspath(path),
        })
    return sorted(records, key=lambda item: item["run_name"])


def main(args):
    records = collect(args.checkpoint_root)
    if not records:
        raise SystemExit("No training_summary.json files found")
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print("Saved:", os.path.abspath(args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--output", default="validation_summaries.csv")
    main(parser.parse_args())
