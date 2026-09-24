"""Train the PIM-backbone / SAFF-PAN detector."""

import argparse
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


ROOT = Path(__file__).resolve().parent


class RGBDoLPDepthTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        self.data["image_mode"] = "rgb_dolp_depth"
        return super().build_dataset(img_path, mode, batch)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "ultralytics/cfg/datasets/pz6qu.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train")
    parser.add_argument("--name", default="pim_saff_pan")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main(args):
    torch.use_deterministic_algorithms(False)
    model = YOLO(ROOT / "ultralytics/cfg/models/11/pz_pim_saff_pan.yaml")
    model.train(
        trainer=RGBDoLPDepthTrainer,
        data=str(args.data),
        cache=args.cache,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer="SGD",
        augment=False,
        pretrained=False,
        amp=args.amp,
        save_period=5,
        project=str(args.project),
        name=args.name,
    )


if __name__ == "__main__":
    main(parse_args())
