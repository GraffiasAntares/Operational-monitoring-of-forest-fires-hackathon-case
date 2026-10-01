import numpy as np
import torch

from ml.bs.augmentation import augment_bs
from ml.bs.baseline import AdaptiveDnbrBaseline, compute_dnbr, evaluate_bs
from ml.bs.losses import WeightedCrossEntropyDiceLoss
from ml.bs.model import CompactUNet
from ml.bs.training import feature_indices


def _layers(dnbr_class: int = 2) -> dict[str, np.ndarray]:
    s2_pre = np.zeros((10, 4, 4), dtype=np.uint16)
    s2_post = np.zeros((10, 4, 4), dtype=np.uint16)
    s2_pre[6], s2_pre[8] = 8000, 2000
    # Generate separated dNBR bins for labels 0..3.
    post_b8a = (7500, 6000, 3500, 1500)[dnbr_class]
    post_b12 = (2500, 4000, 6500, 8500)[dnbr_class]
    s2_post[6], s2_post[8] = post_b8a, post_b12
    return {
        "s2_pre": s2_pre,
        "s2_post": s2_post,
        "s1_pre": np.zeros((2, 4, 4), dtype=np.int16),
        "s1_post": np.zeros((2, 4, 4), dtype=np.int16),
        "aux": np.stack((np.zeros((4, 4), dtype=np.int16), np.zeros((4, 4), dtype=np.int16), np.full((4, 4), 10, dtype=np.int16))),
    }


def test_adaptive_dnbr_baseline_fits_and_scores_per_landcover() -> None:
    samples = [(_layers(label), np.full((4, 4), label, dtype=np.uint8)) for label in range(4)]
    baseline = AdaptiveDnbrBaseline().fit(samples)
    prediction = baseline.predict(_layers(3))
    assert np.all(prediction == 3)
    metrics = evaluate_bs([prediction], [np.full((4, 4), 3, dtype=np.uint8)])
    assert metrics.burn_iou == 1.0
    assert compute_dnbr(_layers(3)).shape == (4, 4)


def test_unet_loss_and_feature_modes_have_expected_shapes() -> None:
    assert len(feature_indices("s2")) == 18
    assert len(feature_indices("s2_aux")) == 31
    assert len(feature_indices("full")) == 74
    model = CompactUNet(31, base_channels=8)
    logits = model(torch.zeros((1, 31, 64, 64)))
    target = torch.zeros((1, 64, 64), dtype=torch.long)
    loss = WeightedCrossEntropyDiceLoss(torch.tensor((1.0, 2.0, 2.0, 3.0)))(logits, target)
    assert logits.shape == (1, 4, 64, 64)
    assert torch.isfinite(loss)


def test_augment_bs_keeps_image_and_target_aligned_shapes() -> None:
    image = np.zeros((74, 8, 8), dtype=np.float32)
    image[40] = np.arange(64, dtype=np.float32).reshape(8, 8)
    target = np.arange(64, dtype=np.uint8).reshape(8, 8)
    augmented_image, augmented_target = augment_bs(image, target, np.random.default_rng(42))
    assert augmented_image.shape == image.shape
    assert augmented_target.shape == target.shape
    assert set(augmented_target.reshape(-1)) == set(target.reshape(-1))
