#!/usr/bin/env bash

GPUS="0"
export CUDA_VISIBLE_DEVICES=$GPUS
python inference.py --auxiliary "fcn"  \
                    --trunk_head "deeplab" \
                    --imgs_path 'data/SKMT/Seg/JPEGImages'\
                    --num_classes 19 \
                    --savedir "./runs/deeplab_resnet_dws_fcn_focal/" \
                    --gpus $GPUS