"""Trainer that connects PCDNet to this repository's 9-channel data loader."""

import csv
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

from .loss import ComputePCDNetLoss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PCDNET_ROOT = Path(__file__).resolve().parents[3] / "compare" / "AAAI24-PCDNet"


def _import_pcdnet(pcdnet_root):
    root = Path(pcdnet_root).resolve()
    if not (root / "models" / "yolo.py").is_file():
        raise FileNotFoundError(f"PCDNet source was not found at {root}")
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from models.yolo import IDetect  # noqa: PLC0415
    from utils.general import box_iou, non_max_suppression, xywh2xyxy  # noqa: PLC0415
    from utils.metrics import ap_per_class  # noqa: PLC0415

    return IDetect, box_iou, non_max_suppression, xywh2xyxy, ap_per_class


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location=device)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def split_polarization_inputs(images):
    """Extract [S0 RGB, AoLP RGB, DoLP RGB] from the local formatted 9-channel tensor.

    The local Format transform reverses all channels. Its output is:
    depth, AoLP_nir, AoLP_rgb, DoLP_nir, DoLP_rgb, S0_nir, S0_B, S0_G, S0_R.
    PCDNet expects three 3-channel domains, so scalar AoLP/DoLP are repeated.
    """
    if images.ndim != 4 or images.shape[1] != 9:
        raise ValueError(f"Expected BCHW input with 9 channels, got {tuple(images.shape)}")
    s0 = images[:, [8, 7, 6]]
    aolp = images[:, 2:3].repeat(1, 3, 1, 1)
    dolp = images[:, 4:5].repeat(1, 3, 1, 1)
    return [s0, aolp, dolp]


def _batch_targets(batch, device):
    batch_index = batch["batch_idx"].view(-1, 1)
    return torch.cat((batch_index, batch["cls"].view(-1, 1), batch["bboxes"]), 1).to(device)


def _reset_detection_head(model, nc, class_names, detect_type):
    detect = model.model[-1]
    if not isinstance(detect, detect_type):
        raise TypeError(f"PCDNet checkpoint ends in {type(detect).__name__}, expected IDetect")
    old_nc, old_no = detect.nc, detect.no
    if old_nc != nc:
        new_no = nc + 5
        new_convs, new_implicit = nn.ModuleList(), nn.ModuleList()
        for old_conv, old_implicit in zip(detect.m, detect.im):
            new_conv = nn.Conv2d(old_conv.in_channels, detect.na * new_no, 1, bias=True)
            with torch.no_grad():
                old_w = old_conv.weight.view(detect.na, old_no, *old_conv.weight.shape[1:])
                old_b = old_conv.bias.view(detect.na, old_no)
                new_w = new_conv.weight.view(detect.na, new_no, *new_conv.weight.shape[1:])
                new_b = new_conv.bias.view(detect.na, new_no)
                new_w[:, :5].copy_(old_w[:, :5])
                new_b[:, :5].copy_(old_b[:, :5])
                common = min(old_nc, nc)
                if common:
                    new_w[:, 5 : 5 + common].copy_(old_w[:, 5 : 5 + common])
                    new_b[:, 5 : 5 + common].copy_(old_b[:, 5 : 5 + common])
            new_convs.append(new_conv)
            new_implicit.append(type(old_implicit)(detect.na * new_no))
        detect.m, detect.im = new_convs, new_implicit
        detect.nc, detect.no = nc, new_no
        model.yaml["nc"] = nc
    model.names = class_names
    return old_nc != nc


def _load_model(checkpoint_path, device, nc, names, pcdnet_root, pretrained=False, resume=False):
    detect_type, *_ = _import_pcdnet(pcdnet_root)
    checkpoint = _torch_load(checkpoint_path, device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a PCDNet checkpoint dictionary containing 'model' or 'ema'")
    template = next(
        (checkpoint.get(key) for key in ("ema", "model") if isinstance(checkpoint.get(key), nn.Module)),
        None,
    )
    if not isinstance(template, nn.Module):
        raise ValueError(
            "This PCDNet release has no architecture YAML, so training requires the author's full-object "
            "PCDNet.pt checkpoint (not a state_dict-only file)."
        )
    if resume or pretrained:
        model = template.float()
    else:
        if not hasattr(template, "yaml"):
            raise ValueError("The architecture checkpoint does not contain PCDNet's model.yaml dictionary")
        # The public repository omits its model YAML. Reconstructing from this embedded dictionary
        # gives a genuinely random initialization without copying any checkpoint parameters.
        model = type(template)(cfg=deepcopy(template.yaml), nc=nc).float()
    changed = _reset_detection_head(model, nc, names, detect_type)
    return model, checkpoint, changed


def _make_loader(data, cfg, split, batch_size, workers, augment):
    mode = "train" if augment else "val"
    dataset = build_yolo_dataset(cfg, data[split], batch_size, data, mode=mode, rect=False, stride=32)
    return build_dataloader(dataset, batch_size, workers, shuffle=split == "train", rank=-1)


def _match_predictions(predictions, labels, iouv, box_iou):
    correct = torch.zeros((predictions.shape[0], iouv.numel()), dtype=torch.bool, device=predictions.device)
    if not len(labels) or not len(predictions):
        return correct
    iou = box_iou(labels[:, 1:], predictions[:, :4])
    matches = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == predictions[:, 5]))
    if matches[0].numel():
        match_data = torch.cat((torch.stack(matches, 1), iou[matches[0], matches[1]][:, None]), 1).cpu().numpy()
        if match_data.shape[0] > 1:
            match_data = match_data[match_data[:, 2].argsort()[::-1]]
            match_data = match_data[np.unique(match_data[:, 1], return_index=True)[1]]
            match_data = match_data[np.unique(match_data[:, 0], return_index=True)[1]]
        match_data = torch.as_tensor(match_data, device=predictions.device)
        correct[match_data[:, 1].long()] = match_data[:, 2:3] >= iouv
    return correct


class PCDNetTrainer:
    """Train the released PCDNet model with the repository's polarization dataset."""

    def __init__(self, args):
        self.args = args
        self.device = self._select_device(args.device)
        self._seed_everything(args.seed)
        self.pcdnet_api = _import_pcdnet(args.pcdnet_root)
        self.data = check_det_dataset(args.data, autodownload=False)
        self.names = self.data["names"]
        self.nc = self.data["nc"]
        self.save_dir = Path(args.project) / args.name
        self.weights_dir = self.save_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        source_checkpoint = args.resume or args.weights
        if not Path(source_checkpoint).is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {source_checkpoint}. Download the author's PCDNet.pt first; "
                "the released repository does not include a model YAML."
            )
        self.model, self.checkpoint, head_changed = _load_model(
            source_checkpoint,
            self.device,
            self.nc,
            self.names,
            args.pcdnet_root,
            pretrained=args.pretrained,
            resume=bool(args.resume),
        )
        self.initialization = "resume" if args.resume else "pretrained" if args.pretrained else "random"
        print(f"PCDNet initialization: {self.initialization} (seed={args.seed})")
        self.model.to(self.device)
        if "," in args.device and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        overrides = {
            "task": "detect",
            "imgsz": args.imgsz,
            "cache": args.cache,
            "single_cls": False,
            "rect": False,
            "classes": None,
            "fraction": 1.0,
            "mosaic": args.mosaic if args.augment else 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
        }
        cfg = get_cfg(overrides=overrides)
        self.train_loader = _make_loader(self.data, cfg, "train", args.batch, args.workers, args.augment)
        self.val_loader = _make_loader(self.data, cfg, "val", args.batch, args.workers, False)

        self.criterion = ComputePCDNetLoss(
            _unwrap(self.model), box=args.box, obj=args.obj, cls=args.cls, anchor_t=args.anchor_t
        )
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay).append(parameter)
        self.optimizer = torch.optim.SGD(
            no_decay, lr=args.lr0, momentum=args.momentum, weight_decay=0.0, nesterov=True
        )
        self.optimizer.add_param_group({"params": decay, "weight_decay": args.weight_decay})
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: args.lrf + (1 - args.lrf) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2,
        )
        self.scaler = amp.GradScaler(enabled=args.amp and self.device.type == "cuda")
        self.start_epoch, self.best_map = 0, -1.0
        if args.resume:
            if head_changed:
                raise ValueError("Cannot resume after changing the checkpoint detection head class count")
            self._restore_training_state()

    @staticmethod
    def _seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _select_device(spec):
        if spec.lower() == "cpu":
            return torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = spec
        if not torch.cuda.is_available():
            print("CUDA is unavailable; falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda:0")

    def _restore_training_state(self):
        if self.checkpoint.get("optimizer"):
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])
        if self.checkpoint.get("scheduler"):
            self.scheduler.load_state_dict(self.checkpoint["scheduler"])
        self.start_epoch = int(self.checkpoint.get("epoch", -1)) + 1
        self.best_map = float(self.checkpoint.get("best_map", -1.0))

    def train(self):
        csv_path = self.save_dir / "results.csv"
        if self.start_epoch == 0:
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(
                    ["epoch", "train_box", "train_obj", "train_cls", "val_loss", "precision", "recall", "map50", "map50_95"]
                )

        for epoch in range(self.start_epoch, self.args.epochs):
            components = self._train_epoch(epoch)
            metrics = self.validate()
            self.scheduler.step()
            row = [epoch, *components, metrics["loss"], metrics["precision"], metrics["recall"], metrics["map50"], metrics["map"]]
            with csv_path.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow([f"{value:.6g}" if isinstance(value, float) else value for value in row])
            is_best = metrics["map"] > self.best_map
            self.best_map = max(self.best_map, metrics["map"])
            self._save(epoch, self.weights_dir / "last.pt")
            if is_best:
                self._save(epoch, self.weights_dir / "best.pt")
            print(
                f"epoch {epoch + 1}/{self.args.epochs} | box {components[0]:.4f} obj {components[1]:.4f} "
                f"cls {components[2]:.4f} | P {metrics['precision']:.4f} R {metrics['recall']:.4f} "
                f"mAP50 {metrics['map50']:.4f} mAP50-95 {metrics['map']:.4f}"
            )

    def _train_epoch(self, epoch):
        self.model.train()
        running = torch.zeros(3, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            images = batch["img"].to(self.device, non_blocking=True).float() / 255.0
            targets = _batch_targets(batch, self.device)
            with amp.autocast(enabled=self.scaler.is_enabled()):
                predictions = self.model(split_polarization_inputs(images))
                loss, parts = self.criterion(predictions, targets)
                loss = loss / self.args.accumulate
            self.scaler.scale(loss).backward()
            if (step + 1) % self.args.accumulate == 0 or step + 1 == len(self.train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            running = (running * step + parts) / (step + 1)
        return tuple(float(x) for x in running.cpu())

    @torch.no_grad()
    def validate(self):
        _, box_iou, non_max_suppression, xywh2xyxy, ap_per_class = self.pcdnet_api
        self.model.eval()
        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
        stats, loss_sum = [], 0.0
        for batch in self.val_loader:
            images = batch["img"].to(self.device, non_blocking=True).float() / 255.0
            targets = _batch_targets(batch, self.device)
            predictions, raw = self.model(split_polarization_inputs(images))
            loss_sum += float(self.criterion(raw, targets)[0]) / images.shape[0]
            detections = non_max_suppression(predictions, conf_thres=0.001, iou_thres=0.65, multi_label=True)
            height, width = images.shape[2:]
            for image_index, detection in enumerate(detections):
                labels = targets[targets[:, 0] == image_index, 1:].clone()
                target_classes = labels[:, 0].tolist()
                if len(labels):
                    labels[:, 1:] *= torch.tensor([width, height, width, height], device=self.device)
                    labels[:, 1:] = xywh2xyxy(labels[:, 1:])
                correct = _match_predictions(detection, labels, iouv, box_iou)
                stats.append((correct.cpu(), detection[:, 4].cpu(), detection[:, 5].cpu(), target_classes))

        precision = recall = map50 = mean_ap = 0.0
        if stats:
            combined = [np.concatenate(values, 0) for values in zip(*stats)]
            if combined[0].shape[0] and combined[3].shape[0]:
                p, r, ap, _, _ = ap_per_class(*combined, v5_metric=True)
                precision, recall = float(p.mean()), float(r.mean())
                map50, mean_ap = float(ap[:, 0].mean()), float(ap.mean())
        return {
            "loss": loss_sum / max(len(self.val_loader), 1),
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map": mean_ap,
        }

    def _save(self, epoch, path):
        checkpoint = {
            "epoch": epoch,
            "best_map": self.best_map,
            "model": deepcopy(_unwrap(self.model)).half(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "data": str(self.args.data),
            "channel_mapping": "S0_rgb + repeated AoLP_rgb + repeated DoLP_rgb",
            "initialization": self.initialization,
            "train_args": vars(self.args),
        }
        torch.save(checkpoint, path)
