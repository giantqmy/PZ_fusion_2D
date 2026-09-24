"""Visualize PIM and SAFF feature maps for a trained 9-channel YOLO model."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.nn.modules.fusion import PIM, SAFF


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("/media/ddc/新加卷/hys/qmy/PZ_8/runs/train/91512/weights/best.pt"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "ultralytics/cfg/datasets/pz6.yaml",
        help="dataset YAML used to locate the train split",
    )
    parser.add_argument("--source", type=Path, help="override the train image list or directory")
    parser.add_argument(
        "--image",
        type=Path,
        help="visualize exactly one base image; overrides --source, e.g. /data/000123.jpg",
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/feature_vis/91512_pim_saff")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=8, help="0 saves every image")
    parser.add_argument("--channels", type=int, default=16, help="number of individual channels to save per branch")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="confidence threshold for displayed prediction boxes; feature maps are unaffected",
    )
    return parser.parse_args()


def resolve_source(args):
    data = yaml_load(args.data)
    root = Path(data.get("path", args.data.parent)).expanduser()
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()
    source = args.source or Path(data[args.split])
    if not source.is_absolute():
        source = root / source
    return source.resolve()


def tensor_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    if isinstance(value, (tuple, list)):
        return tuple(tensor_copy(item) for item in value)
    return value


class FeatureCapture:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        self.handles = []

    def register(self, model):
        counters = {"PIM": 0, "SAFF": 0}
        for name, module in model.named_modules():
            kind = type(module).__name__
            if isinstance(module, PIM):
                kind = "PIM"
            elif isinstance(module, SAFF):
                kind = "SAFF"
            else:
                continue
            index = counters[kind]
            counters[kind] += 1
            key = f"{kind}_{index:02d}"
            self.handles.append(module.register_forward_pre_hook(self._pre_hook(key)))
            self.handles.append(module.register_forward_hook(self._post_hook(key)))
            print(f"Hooked {key}: {name}")
        if not self.handles:
            raise RuntimeError("No PIM or SAFF modules were found in the loaded model")

    def _pre_hook(self, key):
        def hook(_module, inputs):
            self.inputs[key] = tensor_copy(inputs)

        return hook

    def _post_hook(self, key):
        def hook(_module, _inputs, output):
            self.outputs[key] = tensor_copy(output)

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def branch_pair(value):
    if isinstance(value, tuple) and len(value) >= 2:
        return value[0], value[1]
    if isinstance(value, list) and len(value) >= 2:
        return value[0], value[1]
    raise TypeError(f"Expected two branch tensors, got {type(value).__name__}")


def normalize_map(feature):
    feature = feature.abs().mean(0).numpy()
    low, high = np.percentile(feature, [1, 99])
    return np.clip((feature - low) / max(high - low, 1e-12), 0, 1)


def normalize_pair(first, second):
    first_map = first.abs().mean(0).numpy()
    second_map = second.abs().mean(0).numpy()
    values = np.concatenate((first_map.ravel(), second_map.ravel()))
    low, high = np.percentile(values, [1, 99])
    scale = max(high - low, 1e-12)
    return (
        np.clip((first_map - low) / scale, 0, 1),
        np.clip((second_map - low) / scale, 0, 1),
    )


def cosine_similarity(first, second):
    first = first.flatten(1)
    second = second.flatten(1)
    return torch.nn.functional.cosine_similarity(first, second, dim=1).mean().item()


def branch_stats(before, after):
    delta = after - before
    return {
        "before_abs_mean": float(before.abs().mean()),
        "after_abs_mean": float(after.abs().mean()),
        "delta_abs_mean": float(delta.abs().mean()),
        "delta_rel": float(delta.abs().mean() / (before.abs().mean() + 1e-8)),
        "before_std": float(before.std()),
        "after_std": float(after.std()),
    }


def save_branch_figure(output_dir, key, before, after, channels):
    if before.ndim == 4:
        before = before[0]
    if after.ndim == 4:
        after = after[0]
    delta = after - before
    before_map, after_map = normalize_pair(before, after)
    maps = [before_map, after_map, normalize_map(delta)]
    titles = ["before", "after", "abs delta"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, image, title in zip(axes, maps, titles):
        axis.imshow(image, cmap="turbo", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{key} channel mean")
    fig.tight_layout()
    fig.savefig(output_dir / f"{key}_mean.png", dpi=180)
    plt.close(fig)

    count = min(channels, before.shape[0])
    if count <= 0:
        return
    columns = min(4, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    for channel in range(count):
        axis = axes[channel // columns][channel % columns]
        image = normalize_map(delta[channel : channel + 1])
        axis.imshow(image, cmap="magma", vmin=0, vmax=1)
        axis.set_title(f"delta ch {channel}")
        axis.axis("off")
    for index in range(count, rows * columns):
        axes[index // columns][index % columns].axis("off")
    fig.suptitle(f"{key} per-channel change")
    fig.tight_layout()
    fig.savefig(output_dir / f"{key}_channels.png", dpi=180)
    plt.close(fig)


def save_combined_figure(output_dir, features):
    """Save one comparison sheet containing every PIM/SAFF branch."""
    if not features:
        return
    rows = len(features)
    fig, axes = plt.subplots(rows, 3, figsize=(12, max(3.0 * rows, 5.0)), squeeze=False)
    for row, (key, branch, before, after) in enumerate(features):
        if before.ndim == 4:
            before = before[0]
        if after.ndim == 4:
            after = after[0]
        before_map, after_map = normalize_pair(before, after)
        maps = [before_map, after_map, normalize_map(after - before)]
        for column, (axis, image) in enumerate(zip(axes[row], maps)):
            axis.imshow(image, cmap="turbo", vmin=0, vmax=1)
            axis.axis("off")
            if row == 0:
                axis.set_title(("before", "after", "abs delta")[column], fontsize=11)
        axes[row][0].set_ylabel(f"{key} branch{branch}", fontsize=10)
    fig.suptitle("PIM / SAFF feature comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "all_pim_saff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_input_image(result, output_dir):
    image = result.orig_img
    if image is None:
        return
    output_path = output_dir / "input.png"
    if image.ndim == 3 and image.shape[2] >= 3:
        image = image[:, :, :3]
        # getPolarizationImages() already converts S0 RGB from BGR to RGB.
        image = np.asarray(image)
        if not np.isfinite(image).all():
            image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image /= 255.0
            image = np.clip(image, 0.0, 1.0)
        plt.imsave(output_path, image)


def main(args):
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    source = args.image.expanduser().resolve() if args.image else resolve_source(args)
    if not source.exists():
        raise FileNotFoundError(f"Dataset source not found: {source}")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    capture = FeatureCapture()
    capture.register(model.model)

    csv_rows = []
    try:
        stream = model.predict(
            source=str(source),
            stream=True,
            batch=1,
            ch=9,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            verbose=False,
        )
        processed = 0
        for image_index, result in enumerate(stream):
            if args.max_images and processed >= args.max_images:
                break
            if not capture.inputs or not capture.outputs:
                print(f"WARNING: no captured PIM/SAFF features for image {image_index}")
                continue

            image_dir = args.output / f"image_{processed:04d}"
            image_dir.mkdir(parents=True, exist_ok=True)
            save_input_image(result, image_dir)
            try:
                result.save(filename=str(image_dir / "prediction.jpg"))
            except Exception as error:
                print(f"WARNING: prediction image was not saved: {error}")

            combined_features = []
            for key in sorted(capture.inputs):
                before = branch_pair(capture.inputs[key])
                after = branch_pair(capture.outputs[key])
                module_dir = image_dir / key
                module_dir.mkdir(parents=True, exist_ok=True)
                for branch, (before_branch, after_branch) in enumerate(zip(before, after), 1):
                    save_branch_figure(module_dir, f"branch{branch}", before_branch, after_branch, args.channels)
                    combined_features.append((key, branch, before_branch, after_branch))
                    stats = branch_stats(before_branch, after_branch)
                    stats.update(
                        image=processed,
                        module=key,
                        branch=branch,
                        branch_cosine_before=float(cosine_similarity(before[0], before[1])),
                        branch_cosine_after=float(cosine_similarity(after[0], after[1])),
                    )
                    csv_rows.append(stats)
            save_combined_figure(image_dir, combined_features)
            processed += 1
            capture.inputs.clear()
            capture.outputs.clear()
            print(f"Saved feature maps for image {processed}: {image_dir}")
    finally:
        capture.close()

    if not csv_rows:
        raise RuntimeError("No feature maps were saved")
    with (args.output / "feature_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main(parse_args())
