
#!/usr/bin/env bash

GPUS="1"
export CUDA_VISIBLE_DEVICES=$GPUS
python main.py --trunk_head "multiscaleattention"\
              --batch_size 2   \
              --crop_size 512 \
              --image_size 512 \
              --max_epochs 100 \
              --num_classes 11 \
              --lr 0.004  \
              --show_interval 50 \
              --show_val_interval 1 \
              --savedir "./runs/multiScaleAttention_resnet_fcn/" \
              --gpus $GPUS
