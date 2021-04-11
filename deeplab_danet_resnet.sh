
#!/usr/bin/env bash
GPUS="2"
export CUDA_VISIBLE_DEVICES=$GPUS
python main.py --trunk_head "deeplab_danet"\
              --auxiliary "fcn"\
	      --backbone "resnet50" \
              --batch_size 4   \
              --crop_size 512 \
              --image_size 512 \
              --max_epochs 100 \
              --num_classes 3 \
              --lr 0.004  \
              --show_interval 50 \
              --show_val_interval 1 \
              --savedir "./runs/CAMUS/deeplab_nonlocal/" \
              --gpus $GPUS
