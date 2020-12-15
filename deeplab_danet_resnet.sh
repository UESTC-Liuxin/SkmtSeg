
#!/usr/bin/env bash

GPUS="0"
export CUDA_VISIBLE_DEVICES=$GPUS
python main.py --trunk_head "deeplab_danet"\
              --batch_size 4   \
              --crop_size 512 \
              --image_size 512 \
              --max_epochs 200 \
              --num_classes 11 \
              --lr 0.004  \
              --show_interval 50 \
              --show_val_interval 1 \
              --savedir "./runs/danet_resnet_dws_fcn/" \
              --gpus $GPUS
