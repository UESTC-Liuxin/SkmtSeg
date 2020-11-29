#!/usr/bin/env bash

GPUS="0"
export CUDA_VISIBLE_DEVICES=$GPUS
python main.py --batch_size 4   \
              --auxiliary "fcn"  \
              --trunk_head "DLinkNet" \
              --crop_size 512 \
              --image_size 512 \
              --max_epochs 200 \
              --num_classes 19 \
              --lr 0.004  \
              --show_interval 50 \
              --show_val_interval 1 \
              --savedir "./runs/Dlinknet_resnet_dws_fcn/" \
              --gpus $GPUS

