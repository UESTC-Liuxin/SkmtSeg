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
from dataloader import build_dataset, CallDataSet,SkmtDataSet
from utils.summaries import TensorboardSummary
from utils.modeltools import netParams
from utils.set_logger import get_logger

from tools.train import Trainer
from tools.test import Tester
from modeling import build_skmtnet

def main(args,logger,summary):
    cudnn.enabled = True     # Enables bencnmark mode in cudnn, to enable the inbuilt
    cudnn.benchmark = True   # cudnn auto-tuner to find the best algorithm to use for
                             # our hardware

    seed = random.randint(1, 10000)
    # seed =920
    logger.info('======>random seed {}'.format(seed))

    random.seed(seed)  # python random seed
    np.random.seed(seed)  # set numpy random seed
    torch.manual_seed(seed)  # set random seed for cpu
    train_set, val_set = build_dataset(args)
    kwargs = {'num_workers': args.workers, 'pin_memory': True}

    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
    test_loader = DataLoader(val_set, batch_size=1, drop_last=True, shuffle=False, **kwargs)


    logger.info('======> building network')
    # set model
    model = build_skmtnet(backbone=args.backbone, auxiliary_head=args.auxiliary, trunk_head=args.trunk_head,
                          num_classes=args.num_classes, output_stride=16, img_size=args.image_size)

    logger.info("======> computing network parameters")
    total_paramters = netParams(model)
    logger.info("the number of parameters: " + str(total_paramters))

    # # setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # #SGD
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # #momentum
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # #RMSprop
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
    # #Adam
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    # setup savedir
    args.savedir = (args.savedir + '/' + args.model + 'bs'
                    + str(args.batch_size) + 'gpu' + str(args.gpus) + '/')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    # setup optimization criterion
    # , weight = np.array(SkmtDataSet.CLASSES_PIXS_WEIGHTS)
    weight = [0.22762148, 0.22762148, 0.91752577, 0.22791293, 0.22967742, 1.11949686, 0.99441341, 0.72357724,
              0.71485944, 1, 0.71774194]
    # weight = [0.0326274, 0.47834152,0.26023913,0.25890819,0.39852193,2.95347985,2.74947394,14.42599523,
    #           2.89106704,22.28744334,1]
    if(args.auxiliary is not None):
        CRITERION = dict(
            auxiliary=dict(
                losses=dict(
                    ce=dict(reduction='mean')
                    ,dice=dict(smooth=1, p=2, reduction='mean')
                ),
                loss_weights=[0.5, 0.5]
                # loss_weights = [1]
            ),
            trunk=dict(
                losses=dict(
                    ce=dict(reduction='mean')
                    , dice=dict(smooth=1, p=2, reduction='mean')
                ),
                loss_weights=[0.5, 0.5]
                # loss_weights=[0.5,0.5]
                # loss_weights=[1]
            )
        )
    else:
        CRITERION = dict(
            auxiliary=None,
            trunk=dict(
                losses=dict(
                     ce=dict(reduction='mean')
                    ,dice=dict(smooth=1, p=2, reduction='mean')
                    #,focal = dict(reduction='mean', alpha=torch.softmax(torch.Tensor(weight), dim=0))
                ),
                loss_weights=[0.5,0.5]
            )
        )
    criterion = build_criterion(**CRITERION)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # set random seed for all GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        model=model.cuda()
        criterion=criterion.cuda()


    start_epoch = 0
    best_mIoU = 0.0

    trainer = Trainer(args=args,dataloader=train_loader,model=model,
                    optimizer=optimizer,criterion=criterion,logger=logger,summary=summary)
    tester = Tester(args=args,dataloader=test_loader,model=model,
                    criterion=criterion,logger=logger,summary=summary)

    writer=summary.create_summary()
    for epoch in range(start_epoch,args.max_epochs):
        trainer.train_one_epoch(epoch,writer)

        if(epoch%args.show_val_interval==0):
            Acc,mAcc,mIoU,FWIoU,tb_overall,confusion_matrix=tester.test_one_epoch(epoch,writer)

            new_pred = mIoU
            if new_pred > best_mIoU:
                best_mIoU = new_pred
                best_overall = tb_overall
                best_confusion_matrix = confusion_matrix
                # save the confusion matrix
                data1 = pd.DataFrame(best_confusion_matrix)
                data1.to_csv(args.savedir + 'confusion_matrix.csv')

                # save the model
                model_file_name = args.savedir + '/best_model.pth'
                state = {"epoch": epoch + 1,
                         "model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "criterion": criterion.state_dict()
                         }
                torch.save(state, model_file_name)
            logger.info("======>best overall:")
            logger.info(best_overall)
            logger.info("======>best mIoU:")
            logger.info(best_mIoU)

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
    parser.add_argument('--dataset', default='skmt', type=str)
    parser.add_argument('--auxiliary', default=None, type=str)
    parser.add_argument('--trunk_head', default='deeplab', type=str)
    parser.add_argument('--backbone', default=None, type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--deep_supervision', default=False, type=bool)
    parser.add_argument('--max_epochs', type=int, help='the number of epochs: default 100 ')
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--show_interval', default=50, type=int)
    parser.add_argument('--show_val_interval', default=5, type=int)
    parser.add_argument('--savedir', default="./runs", help="directory to save the model snapshot")
    # parser.add_argument('--logFile', default= "log.txt", help = "storing the training and validation logs")
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--resume', default=None, help="the resume model path")
    args = parser.parse_args()

    # 设置运行id
    run_id = 'lr{}_bz{}'.format(args.lr, args.batch_size) \
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