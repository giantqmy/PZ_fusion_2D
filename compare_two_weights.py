"""Compare two 9-channel YOLO checkpoints on the same images and validation split.

Example:
    python compare_two_weights.py \
        --weights runs/train/rgb_dolp/weights/best.pt runs/train/rgb_dolp_depth/weights/best.pt \
        --data ultralytics/cfg/datasets/pz6qu.yaml \
        --split val --image-mode rgb_dolp_depth --max-images 20

The script writes one annotated image per model, a side-by-side comparison image,
prediction JSON, and independent validation metrics for both checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.data.utils import getPolarizationImages
from ultralytics.utils import yaml_load, yaml_save


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
AUXILIARY_SUFFIXES = (
    "_s0_rgb",
    "_s0_nir",
    "_dolp_rgb",
    "_dolp_nir",
    "_aolp_rgb",
    "_aolp_nir",
    "_depth",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        nargs=2,
        required=True,
        type=Path,
        metavar=("WEIGHTS_A", "WEIGHTS_B"),
        help="two .pt checkpoints; the first and second columns use these files respectively",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "ultralytics/cfg/datasets/pz6qu.yaml",
        help="dataset YAML used for the validation split",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--source", type=Path, help="image, image directory, or image-list file for visual comparison")
    parser.add_argument("--image", type=Path, help="compare exactly one base image; overrides --source")
    parser.add_argument(
        "--image-mode",
        choices=("rgb_dolp_depth", "full"),
        default="rgb_dolp_depth",
        help="input channels used by the custom loader; 'full' loads all polarization channels",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs/compare_two_weights")
    parser.add_argument("--labels", nargs=2, metavar=("LABEL_A", "LABEL_B"), help="short names shown on output images")
    parser.add_argument("--device", default="0", help="CUDA device such as 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4, help="validation batch size")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for comparison images")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-images", type=int, default=20, help="0 compares every source image")
    parser.add_argument("--no-plots", action="store_true", help="disable validation curves/confusion plots")
    return parser.parse_args()


def resolve_data_root(data_path: Path, data: dict) -> Path:
    root = Path(data.get("path", data_path.parent)).expanduser()
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    return root


def resolve_split(data_path: Path, split: str, source: Path | None) -> Path:
    data = yaml_load(data_path)
    root = resolve_data_root(data_path, data)
    selected = source or Path(data[split])
    if not selected.is_absolute():
        selected = root / selected
    return selected.expanduser().resolve()


def is_base_image(path: Path) -> bool:
    stem = path.stem.lower()
    return path.suffix.lower() in IMAGE_SUFFIXES and not stem.endswith(AUXILIARY_SUFFIXES)


def read_source_images(source: Path, root: Path) -> list[Path]:
    if source.is_dir():
        paths = sorted(path for path in source.rglob("*") if is_base_image(path))
    elif source.suffix.lower() in IMAGE_SUFFIXES:
        paths = [source]
    else:
        paths = []
        for raw_line in source.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            paths.append(path if path.is_absolute() else (root / path).resolve())
        paths = [path for path in paths if is_base_image(path)]
    return [path for path in paths if path.is_file()]


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "model"


def checkpoint_label(weight: Path, requested: str | None) -> str:
    return safe_name(requested or weight.stem)


def load_input(path: Path, image_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the detector's HWC 9-channel input and an RGB image for drawing."""
    mode = None if image_mode == "full" else image_mode
    channels = getPolarizationImages(str(path), image_mode=mode)
    if channels.ndim != 3 or channels.shape[2] != 9:
        raise ValueError(f"Expected 9 channels for {path}, got {channels.shape}")
    rgb = np.nan_to_num(channels[..., :3], nan=0.0, posinf=255.0, neginf=0.0)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return channels, rgb


def prediction_record(result, path: Path) -> dict:
    boxes = result.boxes
    detections = []
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        conf = boxes.conf.detach().cpu().numpy()
        names = result.names or {}
        for box, class_id, confidence in zip(xyxy, cls, conf):
            name = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else names[int(class_id)]
            detections.append(
                {
                    "class_id": int(class_id),
                    "class_name": str(name),
                    "confidence": float(confidence),
                    "xyxy": [float(value) for value in box],
                }
            )
    return {"image": str(path), "detections": detections}


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    height = max(30, round(image.shape[0] * 0.045))
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (25, 25, 25), -1)
    scale = max(0.45, min(image.shape[:2]) / 1000)
    cv2.putText(image, text, (8, round(height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def save_comparison(
    models: list[YOLO],
    labels: list[str],
    path: Path,
    args,
    output: Path,
    records: list[dict],
):
    channels, rgb = load_input(path, args.image_mode)
    # Results.plot expects an OpenCV/BGR canvas, while the custom loader input is RGB HWC.
    canvas_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    annotated = []
    image_record = {"image": str(path), "models": {}}
    for index, (model, label) in enumerate(zip(models, labels), 1):
        result = model.predict(
            source=channels,
            ch=9,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]
        plotted = result.plot(img=canvas_bgr, conf=True, labels=True, boxes=True)
        plotted = add_header(plotted, f"{label} | {len(result.boxes) if result.boxes is not None else 0} detections")
        annotated.append(plotted)
        image_record["models"][label] = prediction_record(result, path)["detections"]
        model_dir = output / "models" / f"{index}_{label}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{path.stem}.jpg"
        if not cv2.imwrite(str(model_path), plotted):
            raise OSError(f"Failed to save {model_path}")

    comparison = np.concatenate(annotated, axis=1)
    comparison_dir = output / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = comparison_dir / f"{path.stem}.jpg"
    if not cv2.imwrite(str(comparison_path), comparison):
        raise OSError(f"Failed to save {comparison_path}")
    records.append(image_record)
    print(f"Saved comparison: {comparison_path}")


def make_eval_data(data_path: Path, image_mode: str, output: Path) -> Path:
    data = yaml_load(data_path)
    if "path" in data and data["path"]:
        data["path"] = str(resolve_data_root(data_path, data))
    if image_mode == "full":
        data.pop("image_mode", None)
    else:
        data["image_mode"] = image_mode
    eval_data = output / f"eval_data_{image_mode}.yaml"
    yaml_save(eval_data, data)
    return eval_data


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def evaluate_model(model: YOLO, weight: Path, label: str, eval_data: Path, args, output: Path) -> dict:
    eval_root = output / "evaluation"
    metrics = model.val(
        data=str(eval_data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        ch=9,
        project=str(eval_root),
        name=label,
        exist_ok=True,
        plots=not args.no_plots,
        verbose=False,
    )
    values = {key: to_builtin(value) for key, value in metrics.results_dict.items()}
    values.update(
        {
            "weights": str(weight),
            "label": label,
            "split": args.split,
            "image_mode": args.image_mode,
        }
    )
    # The weight path is attached by main after validation, keeping this helper reusable.
    (eval_root / label).mkdir(parents=True, exist_ok=True)
    (eval_root / label / "metrics.json").write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return values


def main(args):
    weights = [path.expanduser().resolve() for path in args.weights]
    for weight in weights:
        if not weight.is_file():
            raise FileNotFoundError(f"Weights not found: {weight}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    labels = [checkpoint_label(weight, requested) for weight, requested in zip(weights, args.labels or [None, None])]
    if labels[0] == labels[1]:
        labels[1] = f"{labels[1]}_b"
    args.output.mkdir(parents=True, exist_ok=True)
    models = [YOLO(str(weight)) for weight in weights]

    data = yaml_load(args.data)
    root = resolve_data_root(args.data, data)
    source = args.image.expanduser().resolve() if args.image else resolve_split(args.data, args.split, args.source)
    if not source.exists():
        raise FileNotFoundError(f"Comparison source not found: {source}")
    image_paths = read_source_images(source, root)
    if args.image:
        image_paths = [source] if is_base_image(source) else []
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No base images found in {source}")

    records = []
    for path in image_paths:
        save_comparison(models, labels, path, args, args.output, records)
    (args.output / "predictions.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    eval_data = make_eval_data(args.data, args.image_mode, args.output)
    metric_rows = []
    for weight, model, label in zip(weights, models, labels):
        values = evaluate_model(model, weight, label, eval_data, args, args.output)
        metric_rows.append(values)

    keys = sorted({key for row in metric_rows for key in row})
    with (args.output / "evaluation_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"Saved {len(image_paths)} comparisons and evaluation metrics to {args.output.resolve()}")


if __name__ == "__main__":
    main(parse_args())
