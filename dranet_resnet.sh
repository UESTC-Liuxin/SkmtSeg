#!/usr/bin/env bash

GPUS="5"
export CUDA_VISIBLE_DEVICES=$GPUS
python main.py --trunk_head "dranet"\
              --backbone "resnet101"\
              --auxiliary "fcn"\
              --batch_size 4   \
              --crop_size 512 \
              --image_size 512 \
              --max_epochs 200 \
              --num_classes 11 \
              --lr 0.004  \
              --show_interval 50 \
              --show_val_interval 1 \
              --savedir "./runs/dranet_resnet_dws_fcn/" \
              --gpus $GPUS