"""YOLOv5/YOLOv7-style anchor loss used by PCDNet's IDetect head."""

import math

import torch
import torch.nn as nn


def _bbox_iou(box1, box2, eps=1e-7):
    """Element-wise CIoU for boxes in xywh format."""
    b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2

    inter = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(0) * (
        torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)
    ).clamp(0)
    w1, h1 = (b1_x2 - b1_x1).clamp(eps), (b1_y2 - b1_y1).clamp(eps)
    w2, h2 = (b2_x2 - b2_x1).clamp(eps), (b2_y2 - b2_y1).clamp(eps)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)
    c2 = cw.square() + ch.square() + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).square() + (b2_y1 + b2_y2 - b1_y1 - b1_y2).square()) / 4
    v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).square()
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)
    return iou - (rho2 / c2 + v * alpha)


def _smooth_bce(eps=0.0):
    return 1.0 - 0.5 * eps, 0.5 * eps


class ComputePCDNetLoss:
    """Compute box, objectness and classification losses for an IDetect head."""

    def __init__(self, model, box=0.05, obj=1.0, cls=0.5, anchor_t=4.0, label_smoothing=0.0):
        device = next(model.parameters()).device
        detect = model.model[-1]
        self.detect = detect
        self.nc = detect.nc
        self.nl = detect.nl
        self.na = detect.na
        self.anchor_t = anchor_t
        self.gains = (box, obj, cls)
        self.cp, self.cn = _smooth_bce(label_smoothing)
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        self.balance = ([4.0, 1.0, 0.4] if self.nl == 3 else [4.0, 1.0, 0.25, 0.06, 0.02])[: self.nl]

    def __call__(self, predictions, targets):
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self._build_targets(predictions, targets)

        for layer_index, prediction in enumerate(predictions):
            image_index, anchor_index, grid_y, grid_x = indices[layer_index]
            target_obj = torch.zeros(prediction.shape[:4], dtype=prediction.dtype, device=device)
            if image_index.numel():
                selected = prediction[image_index, anchor_index, grid_y, grid_x]
                pxy = selected[:, :2].sigmoid() * 2 - 0.5
                pwh = (selected[:, 2:4].sigmoid() * 2).square() * anchors[layer_index]
                iou = _bbox_iou(torch.cat((pxy, pwh), 1), tbox[layer_index])
                lbox += (1.0 - iou).mean()
                target_obj[image_index, anchor_index, grid_y, grid_x] = iou.detach().clamp(0).to(target_obj.dtype)

                if self.nc > 1:
                    target_cls = torch.full_like(selected[:, 5:], self.cn)
                    target_cls[range(image_index.shape[0]), tcls[layer_index]] = self.cp
                    lcls += self.bce_cls(selected[:, 5:], target_cls)

            lobj += self.bce_obj(prediction[..., 4], target_obj) * self.balance[layer_index]

        lbox *= self.gains[0]
        lobj *= self.gains[1]
        lcls *= self.gains[2]
        components = torch.cat((lbox, lobj, lcls))
        return components.sum() * predictions[0].shape[0], components.detach()

    def _build_targets(self, predictions, targets):
        num_targets = targets.shape[0]
        gain = torch.ones(7, device=targets.device)
        anchor_indices = torch.arange(self.na, device=targets.device).float().view(self.na, 1).repeat(1, num_targets)
        expanded = torch.cat((targets.repeat(self.na, 1, 1), anchor_indices[..., None]), 2)
        offset_base = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * 0.5

        tcls, tbox, indices, matched_anchors = [], [], [], []
        for layer_index, prediction in enumerate(predictions):
            anchors = self.detect.anchors[layer_index]
            gain[2:6] = torch.tensor(prediction.shape, device=targets.device)[[3, 2, 3, 2]]
            scaled = expanded * gain
            if num_targets:
                ratios = scaled[..., 4:6] / anchors[:, None]
                scaled = scaled[torch.maximum(ratios, 1 / ratios).amax(2) < self.anchor_t]
                grid_xy = scaled[:, 2:4]
                inverse_xy = gain[[2, 3]] - grid_xy
                x_mask, y_mask = ((grid_xy % 1 < 0.5) & (grid_xy > 1)).T
                ix_mask, iy_mask = ((inverse_xy % 1 < 0.5) & (inverse_xy > 1)).T
                masks = torch.stack((torch.ones_like(x_mask), x_mask, y_mask, ix_mask, iy_mask))
                scaled = scaled.repeat((5, 1, 1))[masks]
                offsets = (torch.zeros_like(grid_xy)[None] + offset_base[:, None])[masks]
            else:
                scaled = expanded[0]
                offsets = 0

            image_cls, grid_xy, grid_wh, anchor_index = scaled.chunk(4, 1)
            anchor_index = anchor_index.long().view(-1)
            image_index, class_index = image_cls.long().T
            grid_ij = (grid_xy - offsets).long()
            grid_x, grid_y = grid_ij.T
            indices.append(
                (image_index, anchor_index, grid_y.clamp_(0, prediction.shape[2] - 1), grid_x.clamp_(0, prediction.shape[3] - 1))
            )
            tbox.append(torch.cat((grid_xy - grid_ij, grid_wh), 1))
            matched_anchors.append(anchors[anchor_index])
            tcls.append(class_index)
        return tcls, tbox, indices, matched_anchors
