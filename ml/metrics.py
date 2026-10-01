"""Compatibility facade for the AF and BS metric packages."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.af.metrics import binary_f1
from ml.bs.metrics import burn_iou, class_iou, severity_miou


@dataclass(frozen=True)
class CompetitionScore:
    f1_af: float
    iou_burn: float
    miou_sev: float

    @property
    def score(self) -> float:
        return 0.35 * self.f1_af + 0.35 * self.iou_burn + 0.30 * self.miou_sev


def competition_score(
    af_prediction: np.ndarray,
    af_target: np.ndarray,
    bs_prediction: np.ndarray,
    bs_target: np.ndarray,
) -> CompetitionScore:
    return CompetitionScore(
        f1_af=binary_f1(af_prediction, af_target),
        iou_burn=burn_iou(bs_prediction, bs_target),
        miou_sev=severity_miou(bs_prediction, bs_target),
    )
