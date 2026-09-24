import cv2
import numpy as np
import pytest
import torch

from ultralytics.data.utils import getPolarizationImages, normalize_polarization_batch


def test_rgb_dolp_depth_reads_only_required_modalities(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    base = images / "sample.png"
    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)  # BGR
    dolp = np.array([[40]], dtype=np.uint8)
    depth = np.array([[512]], dtype=np.uint16)
    expected = {
        str(images / "sample_S0_rgb.png"): (cv2.IMREAD_COLOR, rgb),
        str(images / "sample_dolp_rgb.png"): (cv2.IMREAD_GRAYSCALE, dolp),
        str(depth_dir / "sample_depth_dense_u16.png"): (cv2.IMREAD_UNCHANGED, depth),
    }
    reads = []

    monkeypatch.setattr("ultralytics.data.utils.os.path.exists", lambda path: path in expected)

    def imread(path, flag):
        reads.append(path)
        expected_flag, image = expected[path]
        assert flag == expected_flag
        return image

    monkeypatch.setattr(cv2, "imread", imread)
    image = getPolarizationImages(str(base), image_mode="rgb_dolp_depth")

    assert reads == list(expected)
    assert image.shape == (1, 1, 9)
    np.testing.assert_array_equal(image[0, 0], [30, 20, 10, 0, 40, 0, 0, 0, 512])
    batch = torch.from_numpy(image.transpose(2, 0, 1)[::-1].copy())[None]
    normalized = normalize_polarization_batch(batch)
    assert normalized[0, 0, 0, 0] == pytest.approx((2 - 29.652361724372745) / 23.830449242515485)
    assert normalized[0, 4, 0, 0] == pytest.approx(40 / 255)
    assert normalized[0, 8, 0, 0] == pytest.approx(30 / 255)
