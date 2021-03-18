#!/usr/bin/env bash

GPUS="0"
export CUDA_VISIBLE_DEVICES=$GPUS
python main.py --batch_size 4   \
              --trunk_head "unet_2plus" \
              --backbone "resnet50"  \
              --auxiliary "fcn"  \
              --crop_size 448 \
              --image_size 448  \
              --max_epochs 100 \
              --num_classes 11 \
              --lr 0.004  \
              --show_interval 50 \
              --show_val_interval 1 \
              --savedir "./runs/unet_nested/deep_supervision_1" \
              --gpus $GPUS
