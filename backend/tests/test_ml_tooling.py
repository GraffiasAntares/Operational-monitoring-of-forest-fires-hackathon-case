import csv

import numpy as np
import pytest

from ml._shared import ChipRecord
from ml.af.metrics import binary_f1
from ml.bs.metrics import burn_iou, severity_miou
from ml.bs.split import grouped_split
from ml.metrics import competition_score
from ml.submission import decode_rle, encode_rle, validate_submission, write_submission


def test_rle_round_trip_is_row_major_and_one_based() -> None:
    mask = np.array([[0, 1, 1], [2, 0, 1]], dtype=np.uint8)
    assert encode_rle(mask, 1) == "2 2 6 1"
    assert np.array_equal(decode_rle("2 2 6 1", (2, 3)), mask == 1)


def test_metrics_follow_competition_empty_class_rule() -> None:
    af_target = np.array([[1, 0], [0, 255]], dtype=np.uint8)
    af_prediction = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    bs_target = np.array([[1, 0], [2, 255]], dtype=np.uint8)
    bs_prediction = np.array([[1, 0], [0, 3]], dtype=np.uint8)
    assert binary_f1(af_prediction, af_target) == pytest.approx(2 / 3)
    assert burn_iou(bs_prediction, bs_target) == pytest.approx(1 / 2)
    # Class 3 is absent from both prediction and target after nodata removal,
    # therefore its IoU is one under the competition rule.
    assert severity_miou(bs_prediction, bs_target) == pytest.approx((1 + 0 + 1) / 3)
    assert competition_score(af_prediction, af_target, bs_prediction, bs_target).score == pytest.approx(0.35 * (2 / 3) + 0.35 * .5 + .3 * (2 / 3))


def test_grouped_split_keeps_fire_event_together() -> None:
    records = [
        ChipRecord("BS_a", "bs", {"fire_event_id": "event-1"}),
        ChipRecord("BS_b", "bs", {"fire_event_id": "event-1"}),
        ChipRecord("BS_c", "bs", {"fire_event_id": "event-2"}),
    ]
    split = grouped_split(records, validation_fraction=.5, seed=1)
    assert ({"BS_a", "BS_b"} <= set(split["train"])) != ({"BS_a", "BS_b"} <= set(split["validation"]))


def test_af_and_bs_are_exposed_as_independent_packages() -> None:
    from ml.af import ActiveFireDataset
    from ml.bs import BurnSeverityDataset

    assert ActiveFireDataset.kind == "af"
    assert BurnSeverityDataset.kind == "bs"


def test_submission_matches_template_and_detects_overlap(tmp_path) -> None:
    template = tmp_path / "sample_submission.csv"
    with template.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["chip_id", "class_id", "rle"])
        writer.writeheader()
        writer.writerows([
            {"chip_id": "AF_te_000001", "class_id": "1", "rle": ""},
            {"chip_id": "BS_te_000001", "class_id": "1", "rle": ""},
            {"chip_id": "BS_te_000001", "class_id": "2", "rle": ""},
            {"chip_id": "BS_te_000001", "class_id": "3", "rle": ""},
        ])
    output = tmp_path / "submission.csv"
    write_submission(template, {"AF_te_000001": np.array([[1]], dtype=np.uint8), "BS_te_000001": np.array([[1]], dtype=np.uint8)}, output)
    validate_submission(output, template, af_shape=(1, 1), bs_shape=(1, 1))
    with output.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[2]["rle"] = "1 1"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["chip_id", "class_id", "rle"])
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="overlap"):
        validate_submission(output, template, af_shape=(1, 1), bs_shape=(1, 1))
