
#!/usr/bin/env bash
GPUS="4"
export CUDA_VISIBLE_DEVICES=$GPUS
python main.py --trunk_head "deeplab_danet"\
              --auxiliary "fcn"\
	      --backbone "resnet101" \
              --batch_size 4   \
              --crop_size 512 \
              --image_size 512 \
              --max_epochs 200 \
              --num_classes 11 \
              --lr 0.004  \
              --show_interval 50 \
              --show_val_interval 1 \
              --savedir "./runs/deeplab_danet/" \
              --gpus $GPUS
