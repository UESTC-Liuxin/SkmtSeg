
#!/usr/bin/env bash

GPUS="2"
export CUDA_VISIBLE_DEVICES=$GPUS
python main.py --trunk_head "deeplab_danet"\
              --auxiliary "fcn"  \
              --batch_size 4   \
              --crop_size 512 \
              --image_size 512 \
              --max_epochs 100 \
              --num_classes 11 \
              --lr 0.005  \
              --show_interval 100 \
              --show_val_interval 1 \
              --savedir "./runs/deeplab_danet_ce/" \
              --gpus $GPUS
