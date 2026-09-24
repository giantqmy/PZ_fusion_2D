"""Visualize and diagnose Truck detections on a train or validation split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import yaml_load


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
AUXILIARY_SUFFIXES = ("_s0_rgb", "_s0_nir", "_dolp_rgb", "_dolp_nir", "_aolp_rgb", "_aolp_nir", "_depth")
COLORS = {
    "tp": (45, 190, 70),
    "miss": (40, 40, 230),
    "low_conf": (210, 180, 40),
    "confused": (0, 140, 255),
    "localization": (180, 80, 230),
    "fp": (0, 220, 255),
}


def parse_args():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=project / "runs/train/329/weights/best.pt")
    parser.add_argument("--data", type=Path, default=project / "ultralytics/cfg/datasets/pz6qu.yaml")
    parser.add_argument("--split", choices=("train", "val"), default="val", help="Dataset split selected from data YAML.")
    parser.add_argument("--source", type=Path, help="Override the selected split with an image list or directory.")
    parser.add_argument("--val", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help="Override the dataset root in data YAML.")
    parser.add_argument("--labels-dir", type=Path, help="Override the directory containing YOLO txt labels.")
    parser.add_argument("--output", type=Path, help="Output directory (default: runs/truck_<split>_visualization).")
    parser.add_argument("--truck-id", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="1", help="CUDA device such as 1 or cpu.")
    parser.add_argument("--infer-conf", type=float, default=0.001, help="Low threshold used to retain diagnostic candidates.")
    parser.add_argument("--display-conf", type=float, default=0.25, help="Operating threshold used for box colors and status.")
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=0, help="0 processes every selected image containing Truck GT.")
    return parser.parse_args()


def resolve_split(args):
    data = yaml_load(args.data)
    root = (args.root or Path(data.get("path", args.data.parent))).expanduser()
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()
    split = args.source or args.val or Path(data[args.split])
    if not split.is_absolute():
        split = root / split
    names = data.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    else:
        names = {int(key): value for key, value in names.items()}
    return root, split.resolve(), names


def read_image_list(split: Path, root: Path):
    if split.is_dir():
        paths = sorted(path for path in split.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    else:
        paths = []
        for raw_line in split.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            path = Path(line)
            paths.append(path if path.is_absolute() else (root / path).resolve())
    return [path for path in paths if not path.stem.lower().endswith(AUXILIARY_SUFFIXES)]


def find_label(image_path: Path, root: Path, labels_dir: Path | None):
    candidates = []
    if labels_dir:
        candidates.append(labels_dir / f"{image_path.stem}.txt")
    try:
        relative = image_path.relative_to(root)
        parts = list(relative.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            candidates.append(root.joinpath(*parts).with_suffix(".txt"))
    except ValueError:
        pass
    if image_path.parent.name == "images":
        candidates.append(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")
    candidates.append(image_path.with_suffix(".txt"))
    return next((path for path in candidates if path.exists()), candidates[0])


def load_yolo_boxes(label_path: Path, width: int, height: int, class_id: int):
    boxes = []
    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if len(fields) < 5 or int(float(fields[0])) != class_id:
            continue
        x, y, w, h = map(float, fields[1:5])
        boxes.append([(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def label_has_class(label_path: Path, class_id: int):
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if fields and int(float(fields[0])) == class_id:
            return True
    return False


def box_iou(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    area1 = np.prod(np.clip(boxes1[:, 2:] - boxes1[:, :2], 0, None), axis=1)
    area2 = np.prod(np.clip(boxes2[:, 2:] - boxes2[:, :2], 0, None), axis=1)
    return intersection / np.clip(area1[:, None] + area2[None, :] - intersection, 1e-9, None)


def match_trucks(gt_boxes, pred_boxes, pred_cls, pred_conf, class_id, threshold, match_iou):
    candidate_ids = np.flatnonzero((pred_cls == class_id) & (pred_conf >= threshold))
    matches = {}
    used_gt = set()
    ious = box_iou(pred_boxes[candidate_ids], gt_boxes)
    for row in np.argsort(-pred_conf[candidate_ids]):
        if not len(gt_boxes):
            break
        gt_id = int(np.argmax(ious[row]))
        if ious[row, gt_id] >= match_iou and gt_id not in used_gt:
            matches[gt_id] = (int(candidate_ids[row]), float(ious[row, gt_id]))
            used_gt.add(gt_id)
    return matches


def display_image(image_path: Path):
    candidates = (
        image_path.with_name(f"{image_path.stem}_S0_rgb.png"),
        image_path.with_name(f"{image_path.stem}_RGB_RGB.png"),
        image_path,
    )
    for candidate in candidates:
        if candidate.exists():
            image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
            if image is not None:
                return image
    raise FileNotFoundError(f"Display image not found for: {image_path}")


def draw_box(image, box, color, text, width=2):
    x1, y1, x2, y2 = np.rint(box).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, width, cv2.LINE_AA)
    scale = max(0.45, min(image.shape[:2]) / 1000)
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    text_y = max(text_h + baseline + 2, y1)
    cv2.rectangle(image, (x1, text_y - text_h - baseline - 3), (x1 + text_w + 4, text_y + 2), color, -1)
    cv2.putText(image, text, (x1 + 2, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    root, split, yaml_names = resolve_split(args)
    if args.output is None:
        args.output = Path(__file__).resolve().parent / f"runs/truck_{args.split}_visualization"
    image_paths = read_image_list(split, root)
    if not image_paths:
        raise RuntimeError(f"No images found for the {args.split} split from {split}")

    records = []
    truck_images = []
    missing_labels = 0
    missing_truck_images = 0
    for image_path in image_paths:
        label_path = find_label(image_path, root, args.labels_dir)
        if not label_path.exists():
            missing_labels += 1
            continue
        if not label_has_class(label_path, args.truck_id):
            continue
        try:
            image = display_image(image_path)
        except FileNotFoundError as error:
            missing_truck_images += 1
            print(f"WARNING: skipping Truck sample because its display image is missing: {error}")
            continue
        gt_boxes = load_yolo_boxes(label_path, image.shape[1], image.shape[0], args.truck_id)
        truck_images.append((image_path, label_path, gt_boxes))
    if args.max_images:
        truck_images = truck_images[: args.max_images]
    if not truck_images:
        raise RuntimeError(f"No Truck ground-truth boxes were found in the {args.split} split.")

    args.output.mkdir(parents=True, exist_ok=True)
    image_output = args.output / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    model_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    names = {**yaml_names, **model_names}
    thresholds = sorted(set([args.infer_conf, 0.05, 0.1, args.display_conf, 0.5]))
    threshold_hits = {threshold: 0 for threshold in thresholds}
    total_gt = 0

    for image_number, (image_path, label_path, gt_boxes) in enumerate(truck_images, 1):
        result = model.predict(
            source=str(image_path), ch=9, imgsz=args.imgsz, device=args.device,
            conf=args.infer_conf, iou=args.nms_iou, verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            pred_boxes = np.empty((0, 4), dtype=np.float32)
            pred_cls = np.empty(0, dtype=np.int32)
            pred_conf = np.empty(0, dtype=np.float32)
        else:
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_cls = result.boxes.cls.int().cpu().numpy()
            pred_conf = result.boxes.conf.cpu().numpy()

        matches_by_threshold = {
            threshold: match_trucks(
                gt_boxes, pred_boxes, pred_cls, pred_conf, args.truck_id, threshold, args.match_iou
            )
            for threshold in thresholds
        }
        for threshold, matches in matches_by_threshold.items():
            threshold_hits[threshold] += len(matches)
        matches = matches_by_threshold[args.display_conf]
        low_matches = matches_by_threshold[args.infer_conf]
        total_gt += len(gt_boxes)
        canvas = display_image(image_path)
        all_ious = box_iou(pred_boxes, gt_boxes)

        for gt_id, gt_box in enumerate(gt_boxes):
            status, detail, confidence, best_iou = "miss", "no matching prediction", 0.0, 0.0
            color = COLORS["miss"]
            if gt_id in matches:
                pred_id, best_iou = matches[gt_id]
                confidence = float(pred_conf[pred_id])
                status, detail, color = "tp", f"Truck {confidence:.2f}", COLORS["tp"]
            elif gt_id in low_matches:
                pred_id, best_iou = low_matches[gt_id]
                confidence = float(pred_conf[pred_id])
                status, detail, color = "low_conf", f"low-conf Truck {confidence:.3f}", COLORS["low_conf"]
            elif len(pred_boxes):
                overlapping = np.flatnonzero((pred_conf >= args.display_conf) & (all_ious[:, gt_id] >= args.match_iou))
                if len(overlapping):
                    pred_id = int(overlapping[np.argmax(pred_conf[overlapping])])
                    confidence, best_iou = float(pred_conf[pred_id]), float(all_ious[pred_id, gt_id])
                    predicted_name = names.get(int(pred_cls[pred_id]), str(int(pred_cls[pred_id])))
                    status, detail, color = "confused", f"predicted as {predicted_name} {confidence:.2f}", COLORS["confused"]
                    draw_box(canvas, pred_boxes[pred_id], color, f"Pred {predicted_name} {confidence:.2f}")
                else:
                    truck_ids = np.flatnonzero((pred_cls == args.truck_id) & (pred_conf >= args.display_conf))
                    if len(truck_ids):
                        pred_id = int(truck_ids[np.argmax(all_ious[truck_ids, gt_id])])
                        confidence, best_iou = float(pred_conf[pred_id]), float(all_ious[pred_id, gt_id])
                        if best_iou >= 0.1:
                            status, detail, color = "localization", f"poor IoU Truck {confidence:.2f}", COLORS["localization"]
            draw_box(canvas, gt_box, color, f"GT Truck | {detail}", width=3)
            records.append({
                "image": str(image_path), "label": str(label_path), "gt_index": gt_id,
                "status": status, "prediction": detail, "confidence": f"{confidence:.6f}", "iou": f"{best_iou:.6f}",
            })

        matched_pred_ids = {pred_id for pred_id, _ in matches.values()}
        visible_trucks = np.flatnonzero((pred_cls == args.truck_id) & (pred_conf >= args.display_conf))
        for pred_id in visible_trucks:
            if int(pred_id) not in matched_pred_ids:
                draw_box(canvas, pred_boxes[pred_id], COLORS["fp"], f"Truck FP {pred_conf[pred_id]:.2f}")

        status_counts = {key: sum(row["status"] == key for row in records if row["image"] == str(image_path)) for key in COLORS}
        header = f"{image_path.name} | GT {len(gt_boxes)} TP {status_counts['tp']} MISS {len(gt_boxes) - status_counts['tp']}"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (25, 25, 25), -1)
        cv2.putText(canvas, header, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        output_path = image_output / f"{image_number:03d}_{image_path.stem}.jpg"
        if not cv2.imwrite(str(output_path), canvas):
            raise OSError(f"Failed to write {output_path}")
        print(f"[{image_number}/{len(truck_images)}] {image_path.name}: {status_counts}")

    with (args.output / "truck_instances.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    status_totals = {key: sum(row["status"] == key for row in records) for key in COLORS}
    summary_lines = [
        f"weights: {args.weights}", f"dataset split: {args.split}", f"source: {split}",
        f"truck images: {len(truck_images)}", f"truck instances: {total_gt}",
        f"entries skipped for missing labels: {missing_labels}",
        f"Truck images skipped because image files are missing: {missing_truck_images}",
        f"display threshold: {args.display_conf}", f"match IoU: {args.match_iou}",
        "", "status at display threshold:",
        *[f"  {key}: {value}" for key, value in status_totals.items() if key != "fp"],
        "", "Truck recall by confidence threshold:",
        *[f"  conf >= {threshold:g}: {threshold_hits[threshold]}/{total_gt} = {threshold_hits[threshold] / total_gt:.4f}" for threshold in thresholds],
        "", "colors: green=TP, red=miss, cyan=low confidence, orange=class confusion, purple=poor localization, yellow=Truck FP",
    ]
    (args.output / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary_lines))
    print(f"\nSaved diagnostic images and tables to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
