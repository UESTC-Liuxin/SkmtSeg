#!/usr/bin/env bash
GPUS="1"
export CUDA_VISIBLE_DEVICES=$GPUS
python main.py --batch_size 2   \
              --auxiliary "None"  \
              --trunk_head "deeplab" \
              --crop_size 512 \
              --image_size 512 \
              --max_epochs 200 \
              --num_classes 11 \
              --lr 0.004  \
              --show_interval 50 \
              --show_val_interval 1 \
              --savedir "./runs/deeplab_resnet_dws_fcn_focal/" \
              --gpus $GPUS
