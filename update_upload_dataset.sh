#切换环境
source activate deeplab

#合并数据集
root = '/media/Program/CV/Project/SkmtSeg/'
python /media/Program/CV/Project/SkmtSeg/tools/convert_dataset/contact_coco.py

#随机切分
python /media/Program/CV/Project/SkmtSeg/tools/convert_datasets/random_split.py

#上传

scp -r /media/Program/CV/dataset/SKMT/Seg liuxin@192.168.1.110:/home/liuxin/Documents/SkmtSeg/data/SKMT/

