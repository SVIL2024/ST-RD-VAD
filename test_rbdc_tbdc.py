import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from rbdc_tbdc_metrics import (
    _auc_until_one,
    _component_matches,
    evaluate_rbdc_tbdc,
    frame_scores_from_maps,
    load_bbox_annotations,
)


def test_component_matching():
    score = np.zeros((8, 8), dtype=np.float32)
    score[2:5, 2:5] = 1.0
    boxes = [{"track_id": 0, "bbox": (2.0, 2.0, 5.0, 5.0)}]
    matches, components, false_positives = _component_matches(
        score, 0.5, boxes, 0.1
    )
    assert matches == [0]
    assert components == {1}
    assert false_positives == 0


def test_filename_mapping_and_metrics():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        frames = root / "frames" / "Test001"
        annotations = root / "annotations" / "Ped2"
        frames.mkdir(parents=True)
        annotations.mkdir(parents=True)
        frame_paths = []
        for frame_number in range(1, 5):
            path = frames / ("{:03d}.tif".format(frame_number))
            Image.fromarray(np.zeros((20, 40), dtype=np.uint8)).save(str(path))
            frame_paths.append(str(path))
        (annotations / "Test001_gt.txt").write_text(
            "002.jpg 0 20 10 8 8\n003.jpg 0 20 10 8 8\n",
            encoding="utf-8",
        )
        videos = {
            "Test001": {
                "frame": frame_paths,
                "length": len(frame_paths),
            }
        }
        loaded, audit = load_bbox_annotations(
            str(root / "annotations"), "ped2", videos, map_size=8
        )
        assert sorted(loaded["Test001"]["frames"]) == [1, 2]
        assert audit["total_boxes"] == 2
        assert audit["total_tracks"] == 1

        maps = np.zeros((4, 8, 8), dtype=np.float32)
        maps[1:3, 2:6, 3:5] = 1.0
        metrics = evaluate_rbdc_tbdc(
            {"Test001": maps},
            loaded,
            threshold_count=4,
            iou_threshold=0.1,
            track_fraction=0.1,
        )
        assert metrics["total_test_frames"] == 4
        assert metrics["total_region_instances"] == 2
        assert metrics["total_tracks"] == 1
        assert 0.0 <= metrics["rbdc_auc"] <= 1.0
        assert 0.0 <= metrics["tbdc_auc"] <= 1.0


def test_shanghai_zero_based_xyxy_tracks():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        frames = root / "frames" / "01_0014"
        annotations = root / "annotations" / "ShanghaiTech"
        frames.mkdir(parents=True)
        annotations.mkdir(parents=True)
        frame_paths = []
        for frame_index in range(4):
            path = frames / ("{:03d}.jpg".format(frame_index + 1))
            Image.fromarray(np.zeros((20, 40), dtype=np.uint8)).save(str(path))
            frame_paths.append(str(path))
        (annotations / "01_0014.txt").write_text(
            "0,0,10,5,30,15\n0,1,10,5,30,15\n",
            encoding="utf-8",
        )
        videos = {
            "01_0014": {
                "frame": frame_paths,
                "length": len(frame_paths),
            }
        }
        loaded, audit = load_bbox_annotations(
            str(root / "annotations"), "shanghai", videos, map_size=8
        )
        assert sorted(loaded["01_0014"]["frames"]) == [0, 1]
        first_box = loaded["01_0014"]["frames"][0][0]["bbox"]
        assert np.allclose(first_box, [2.0, 2.0, 6.0, 6.0])
        assert audit["total_boxes"] == 2
        assert audit["total_tracks"] == 1
        assert audit["annotation_format"].startswith("track_id,frame_idx")


def test_merl_auc_rule():
    auc = _auc_until_one(
        [2.0, 1.0, 0.5, 0.0],
        [1.0, 0.8, 0.6, 0.0],
    )
    assert abs(auc - 0.5) < 1e-12
    closed_auc = _auc_until_one(
        [0.8, 0.4, 0.0],
        [0.9, 0.5, 0.0],
    )
    assert abs(closed_auc - 0.57) < 1e-12


def test_frame_scores_share_final_maps():
    maps = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 5.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    scores = frame_scores_from_maps({"Test001": maps}, topk_ratio=0.5)
    assert np.allclose(scores["Test001"], [3.5, 5.0])


def main():
    test_component_matching()
    test_filename_mapping_and_metrics()
    test_shanghai_zero_based_xyxy_tracks()
    test_merl_auc_rule()
    test_frame_scores_share_final_maps()
    print("RBDC/TBDC four-metric self-test passed: 5/5")


if __name__ == "__main__":
    main()
