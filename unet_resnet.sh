

#!/usr/bin/env bash

GPUS="2"
export CUDA_VISIBLE_DEVICES=$GPUS
python main.py --batch_size 4   \
              --auxiliary "fcn"  \
              --trunk_head "resunet" \
              --crop_size 512 \
              --image_size 512 \
              --max_epochs 100 \
              --num_classes 4 \
              --lr 0.004  \
              --show_interval 50 \
              --show_val_interval 1 \
              --savedir "./runs/CAMUS/unet/" \
              --gpus $GPUS

