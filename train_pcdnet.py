"""Train AAAI24 PCDNet with this repository's polarization data loader."""

import argparse
from pathlib import Path

from pcdnet_trainer.trainer import DEFAULT_PCDNET_ROOT, PCDNetTrainer


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_PCDNET_ROOT / "ckpt" / "PCDNet.pt",
        help="full PCDNet checkpoint used as an architecture template; parameters are random by default",
    )
    parser.add_argument("--resume", type=Path, default=None, help="resume from this trainer's last.pt")
    parser.add_argument("--pretrained", action="store_true", help="initialize from checkpoint parameters instead of random weights")
    parser.add_argument("--pcdnet-root", type=Path, default=DEFAULT_PCDNET_ROOT)
    parser.add_argument("--data", type=Path, default=ROOT / "ultralytics" / "cfg" / "datasets" / "pz6.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0", help="CUDA device(s), e.g. 0 or 0,1; use cpu for CPU")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "train")
    parser.add_argument("--name", default="pcdnet_s0_aolp_dolp")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--augment", action="store_true", help="enable geometry/mosaic augmentation (AoLP flips stay disabled)")
    parser.add_argument("--mosaic", type=float, default=1.0)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--box", type=float, default=0.05)
    parser.add_argument("--obj", type=float, default=1.0)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--anchor-t", type=float, default=4.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.accumulate < 1:
        parser.error("epochs, batch and accumulate must be positive")
    return args


if __name__ == "__main__":
    PCDNetTrainer(parse_args()).train()
