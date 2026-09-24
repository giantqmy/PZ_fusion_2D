"""Evaluate a trained PCDNet checkpoint on the validation split."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset

from pcdnet_trainer.trainer import (
    DEFAULT_PCDNET_ROOT,
    _batch_targets,
    _import_pcdnet,
    _load_model,
    _match_predictions,
    _make_loader,
    split_polarization_inputs,
)


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "runs" / "train" / "329" / "weights" / "best.pt",
        help="trained PCDNet checkpoint, normally weights/best.pt",
    )
    parser.add_argument("--pcdnet-root", type=Path, default=DEFAULT_PCDNET_ROOT)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "ultralytics" / "cfg" / "datasets" / "pz6.yaml",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0", help="CUDA device, or cpu")
    parser.add_argument("--conf", type=float, default=0.001, help="confidence threshold before AP calculation")
    parser.add_argument("--iou", type=float, default=0.65, help="NMS IoU threshold")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "val")
    parser.add_argument("--name", default="pcdnet_best")
    return parser.parse_args()


def select_device(spec):
    if spec.lower() == "cpu":
        return torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = spec
    if not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU")
        return torch.device("cpu")
    return torch.device("cuda:0")


def class_name(names, index):
    if isinstance(names, dict):
        return str(names.get(index, index))
    return str(names[index]) if index < len(names) else str(index)


def evaluate(args):
    device = select_device(args.device)
    data = check_det_dataset(args.data, autodownload=False)
    names = data["names"]
    nc = data["nc"]
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    api = _import_pcdnet(args.pcdnet_root)
    detect_type, box_iou, non_max_suppression, xywh2xyxy, ap_per_class = api
    model, _, _ = _load_model(
        args.weights,
        device,
        nc,
        names,
        args.pcdnet_root,
        pretrained=True,
    )
    model.to(device).eval()

    cfg = get_cfg(
        overrides={
            "task": "detect",
            "imgsz": args.imgsz,
            "cache": args.cache,
            "single_cls": False,
            "rect": False,
            "classes": None,
            "fraction": 1.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
        }
    )
    loader = _make_loader(data, cfg, args.split, args.batch, args.workers, augment=False)
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats = []
    image_count = 0
    target_count = np.zeros(nc, dtype=np.int64)
    target_images = np.zeros(nc, dtype=np.int64)
    preprocess_time = inference_time = nms_time = 0.0

    with torch.no_grad():
        for batch in loader:
            start = time.perf_counter()
            images = batch["img"].to(device, non_blocking=True).float() / 255.0
            targets = _batch_targets(batch, device)
            preprocess_time += time.perf_counter() - start

            start = time.perf_counter()
            predictions, _ = model(split_polarization_inputs(images))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_time += time.perf_counter() - start

            start = time.perf_counter()
            detections = non_max_suppression(
                predictions,
                conf_thres=args.conf,
                iou_thres=args.iou,
                multi_label=True,
            )
            nms_time += time.perf_counter() - start

            height, width = images.shape[2:]
            for image_index, detection in enumerate(detections):
                labels = targets[targets[:, 0] == image_index, 1:].clone()
                target_classes = labels[:, 0].long().tolist()
                image_count += 1
                for cls in set(target_classes):
                    if 0 <= cls < nc:
                        target_images[cls] += 1
                if len(labels):
                    for cls in labels[:, 0].long().tolist():
                        if 0 <= cls < nc:
                            target_count[cls] += 1
                    labels[:, 1:] *= torch.tensor([width, height, width, height], device=device)
                    labels[:, 1:] = xywh2xyxy(labels[:, 1:])
                correct = _match_predictions(detection, labels, iouv, box_iou)
                stats.append((correct.cpu(), detection[:, 4].cpu(), detection[:, 5].cpu(), target_classes))

    output_dir = args.project / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_results(stats, target_count, target_images, names, nc, ap_per_class)
    result["all"]["images"] = image_count
    result["images"] = image_count
    result["weights"] = str(args.weights.resolve())
    result["data"] = str(args.data.resolve())
    result["split"] = args.split
    result["speed_ms_per_image"] = {
        "preprocess": preprocess_time * 1000 / max(image_count, 1),
        "inference": inference_time * 1000 / max(image_count, 1),
        "postprocess": nms_time * 1000 / max(image_count, 1),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", "images", "instances", "precision", "recall", "mAP50", "mAP50-95"])
        for row in result["per_class"]:
            writer.writerow([row["class"], row["images"], row["instances"], row["precision"], row["recall"], row["map50"], row["map50_95"]])

    print_table(result)
    print(f"\nResults saved to {output_dir}")


def build_results(stats, target_count, target_images, names, nc, ap_per_class):
    precision = recall = map50 = map5095 = 0.0
    per_class = [
        {
            "class": class_name(names, index),
            "images": int(target_images[index]),
            "instances": int(target_count[index]),
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
        }
        for index in range(nc)
    ]
    if stats:
        combined = [np.concatenate(values, 0) for values in zip(*stats)]
        if combined[0].shape[0] and combined[3].shape[0]:
            p, r, ap, _, ap_classes = ap_per_class(*combined, v5_metric=True)
            precision = float(p.mean())
            recall = float(r.mean())
            map50 = float(ap[:, 0].mean())
            map5095 = float(ap.mean())
            for row_index, cls in enumerate(ap_classes.tolist()):
                if 0 <= cls < nc:
                    per_class[cls]["precision"] = float(p[row_index])
                    per_class[cls]["recall"] = float(r[row_index])
                    per_class[cls]["map50"] = float(ap[row_index, 0])
                    per_class[cls]["map50_95"] = float(ap[row_index].mean())
    return {
        "all": {
            "images": int(target_images.sum()),
            "instances": int(target_count.sum()),
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map5095,
        },
        "per_class": per_class,
    }


def print_table(result):
    print("\nClass                 Images  Instances  Box(P)      R     mAP50  mAP50-95")
    overall = result["all"]
    print(
        f"all                   {overall['images']:6d}  {overall['instances']:9d}  "
        f"{overall['precision']:.3f}  {overall['recall']:.3f}  "
        f"{overall['map50']:.3f}    {overall['map50_95']:.3f}"
    )
    for row in result["per_class"]:
        print(
            f"{row['class'][:20]:20s}  {row['images']:6d}  {row['instances']:9d}  "
            f"{row['precision']:.3f}  {row['recall']:.3f}  "
            f"{row['map50']:.3f}    {row['map50_95']:.3f}"
        )
    speed = result["speed_ms_per_image"]
    print(
        f"Speed: {speed['preprocess']:.1f}ms preprocess, {speed['inference']:.1f}ms inference, "
        f"{speed['postprocess']:.1f}ms postprocess per image"
    )


if __name__ == "__main__":
    evaluate(parse_args())
