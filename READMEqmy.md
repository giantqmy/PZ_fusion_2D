```shell
python -u train_pim_saff_pan.py \
  --data "/media/ddc/新加卷/hys/qmy/PZ_8/ultralytics/cfg/datasets/pz6.yaml" \
  --device 0 \
  --batch 16 \
  --epochs 200 \
  --workers 8 \
  --name pim_saff_pan_depth
  ```