# PCDNet training with the local polarization loader

`train_pcdnet.py` reuses this repository's `YOLODataset`, including its image-list and YOLO-label handling. It reads the
existing 9-channel tensor and sends only these signals to PCDNet:

- `S0_rgb`: converted back to RGB and used as PCDNet's first 3-channel domain.
- `AoLP_rgb`: one channel repeated three times for PCDNet's AoLP domain.
- `DoLP_rgb`: one channel repeated three times for PCDNet's DoLP domain.

NIR polarization and depth channels are not sent to the model.

The public PCDNet repository contains neither a training script nor its model YAML. Download the author's full-object
`PCDNet.pt` checkpoint and place it at `compare/AAAI24-PCDNet/ckpt/PCDNet.pt`, or pass its location with `--weights`.
By default this checkpoint is used **only as an architecture template**: the embedded model dictionary is used to build
a new model, and all trainable parameters are randomly initialized with the selected `--seed` (default `0`). No author
checkpoint parameters are copied. This is the appropriate mode when the other comparison models are also trained from
scratch.

```bash
python train_pcdnet.py \
  --weights ../../compare/AAAI24-PCDNet/ckpt/PCDNet.pt \
  --data ultralytics/cfg/datasets/pz6.yaml \
  --device 0 --batch 8 --imgsz 512 --epochs 200
```

Checkpoints and `results.csv` are written under `runs/train/pcdnet_s0_aolp_dolp`. Resume with:

```bash
python train_pcdnet.py --resume runs/train/pcdnet_s0_aolp_dolp/weights/last.pt
```

To run a separate transfer-learning experiment with the author's parameters, add `--pretrained`. Do not mix results
from random initialization and pretrained initialization in the same comparison table unless they are clearly labeled.

Augmentation is off by default. `--augment` enables shared geometric/mosaic transforms, while HSV changes and flips
remain disabled so that scalar AoLP values are not corrupted.
