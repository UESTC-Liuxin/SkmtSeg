#!/usr/bin/env bash
GPUS="6"
export CUDA_VISIBLE_DEVICES=$GPUS
python main.py --batch_size 4   \
              --trunk_head "unet_3plus" \
              --crop_size 418 \
              --image_size 418  \
              --max_epochs 100 \
              --num_classes 11 \
              --lr 0.004  \
              --show_interval 50 \
              --show_val_interval 1 \
              --savedir "./runs/unet_nested3" \
              --gpus $GPUS

