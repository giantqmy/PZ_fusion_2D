```shell
python -u train_pim_saff_pan.py \
  --data "/media/lab3102/12T硬盘/personal_data/qmy/PZ_fusion_2D/ultralytics/cfg/datasets/pz6.yaml" \
  --device 0 \
  --batch 16 \
  --epochs 200 \
  --workers 8 \
  --name pim_saff_pan_depth
  ```