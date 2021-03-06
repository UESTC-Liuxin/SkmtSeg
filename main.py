# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/1 下午10:20
"""
import os
import random
import numpy as np
import argparse
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from criterion import build_criterion
from utils.summaries import TensorboardSummary
from utils.modeltools import netParams
from utils.set_logger import get_logger

from tools.train import Trainer
from tools.test import Tester

from dataloader.skmt import SkmtDataSet
from dataloader.simplers import CustomRandomSampler, BatchSampler
from modeling import build_skmtnet

def main(args,logger,summary):
    seed=6000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_set = SkmtDataSet(args,split='train')
    val_set = SkmtDataSet(args, split='val')
    kwargs = {'num_workers': args.workers, 'pin_memory': True}

    sampler=CustomRandomSampler(train_set,batch_size=args.batch_size)
    batch_sampler=BatchSampler(sampler)

    def worker_init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(train_set, batch_sampler=batch_sampler,worker_init_fn=worker_init_fn,**kwargs)

    test_loader = DataLoader(val_set, batch_size=1, drop_last=True, worker_init_fn=worker_init_fn,shuffle=False, **kwargs)


    logger.info('======> building network')
    # set model
    model = build_skmtnet(backbone='wide_resnet50_2',auxiliary_head=args.auxiliary, trunk_head=args.trunk_head,
                          num_classes=args.num_classes,output_stride = 32)

    logger.info("======> computing network parameters")
    total_paramters = netParams(model)
    logger.info("the number of parameters: " + str(total_paramters))

    # setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # setup savedir
    args.savedir = (args.savedir + '/' + args.model + 'bs'
                    + str(args.batch_size) + 'gpu' + str(args.gpus) + '/')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    # setup optimization criterion
    # , weight = np.array(SkmtDataSet.CLASSES_PIXS_WEIGHTS)
    CRITERION = dict(
        auxiliary=dict(
            losses=dict(
                # smoothce=dict(size_average=True),
                # iou=dict(n_classes=11)
                ce=dict(reduction='mean')
                # dice=dict(smooth=1, p=2, reduction='mean')
            ),
            loss_weights=[1]
        ),

        trunk=dict(
            losses=dict(
                # smoothce=dict(size_average=True),
                # iou=dict(n_classes=11)
                # focal=dict(reduction='mean')
                ce=dict(reduction='mean')
                # dice=dict(smooth=1, p=2, reduction='mean')
            ),
            trunk=dict(
                losses=dict(
                    ce=dict(reduction='mean')
                    # dice=dict(smooth=1, p=2, reduction='mean')
                ),
                loss_weights=[1]
            )
        )
    )
    criterion = build_criterion(**CRITERION)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # set random seed for all GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        model=model.cuda()
        criterion=criterion.cuda()


    start_epoch = 0
    best_mIoU = 0.

    trainer = Trainer(args=args,dataloader=train_loader,model=model,
                    optimizer=optimizer,criterion=criterion,logger=logger,summary=summary)
    tester = Tester(args=args,dataloader=test_loader,model=model,
                    criterion=criterion,logger=logger,summary=summary)

    writer=summary.create_summary()
    for epoch in range(start_epoch,args.max_epochs):
        trainer.train_one_epoch(epoch,writer,best_mIoU)

        if(epoch%1==0):
            Acc,mAcc,mIoU,FWIoU,tb_overall=tester.test_one_epoch(epoch,writer)

            new_pred = mIoU
            if new_pred > best_mIoU:
                best_mIoU = new_pred
                best_overall =tb_overall
                # save the model
                model_file_name = args.savedir + '/best_model.pth'
                state = {"epoch": epoch + 1,
                         "model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "criterion": criterion.state_dict()
                         }
                torch.save(state, model_file_name)
            logger.info("======>best epoch:")
            logger.info(best_overall)

    model_file_name = args.savedir + '/resume_model.pth'
    state = {"epoch": epoch + 1,
             "model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "criterion": criterion.state_dict()
             }
    torch.save(state, model_file_name)


if __name__ == '__main__':

    import timeit
    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description='Semantic Segmentation...')

    parser.add_argument('--model', default='skmtnet', type=str)
    parser.add_argument('--auxiliary', default=None, type=str)
    parser.add_argument('--trunk_head', default='deeplab', type=str)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--max_epochs', type=int, help='the number of epochs: default 100 ')
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--show_interval', default=50, type=int)
    parser.add_argument('--show_val_interval', default=1, type=int)
    parser.add_argument('--savedir', default="./runs", help="directory to save the model snapshot")
    # parser.add_argument('--logFile', default= "log.txt", help = "storing the training and validation logs")
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--resume', default=None, help="the resume model path")
    args = parser.parse_args()

    # 设置运行id
    run_id = 'lr{}_bz{}_wd{}'.format(args.lr, args.batch_size,args.weight_decay) \
             + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')  # 现在

    args.savedir = os.path.join(args.savedir, str(run_id))

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    logger = get_logger(args.savedir)
    logger.info('just do it')
    logger.info('Now run_id {}'.format(run_id))

    if (args.resume):
        if not os.path.exists(args.resume):
            raise Exception("the path of resume is empty!!")

    # 设置tensorboard
    summary = TensorboardSummary(args.savedir)

    logger.info('======>Input arguments:')
    for key, val in vars(args).items():
        logger.info('======> {:16} {}'.format(key, val))

    # 开始运行.........
    main(args, logger, summary)
    end = timeit.default_timer()
    logger.info("training time:{:.4f}".format(1.0 * (end - start) / 3600))
    logger.info('model save in {}.'.format(run_id))




