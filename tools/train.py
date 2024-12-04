# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--wflw65', help='model parameters', action='store_true')

    args = parser.parse_args()
    update_config(config, args)     # 通过给定的yaml文件进行config的rewrite
    return args


def main():

    args = parse_args()
    # config65 = config.clone()
    if args.wflw65:
        config.defrost()  # 解锁
        config.MODEL.NUM_JOINTS = 65    # 因为输出是65个节点
        # config.TRAIN.END_EPOCH = 60    # 先少训练一点
        config.WFLW65 = True
        config.freeze()  # 锁住

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))     # 打印参数到控制台和log文件

    cudnn.benchmark = config.CUDNN.BENCHMARK    # 自动寻找最适合当前配置的高效算法
    cudnn.determinstic = config.CUDNN.DETERMINISTIC     # 启用后，操作将使用可用的确定性算法
    cudnn.enabled = config.CUDNN.ENABLED

    model = models.get_face_alignment_net(config)   # 初始化一个模型，用yaml文件初始化config，用pretrained model初始化model

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # loss
    criterion = torch.nn.MSELoss(size_average=True).cuda()

    optimizer = utils.get_optimizer(config, model)
    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'latest.pth')
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_nme = checkpoint['best_nme']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    dataset_type = get_dataset(config)

    train_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        # num_workers=1,
        pin_memory=config.PIN_MEMORY)

    val_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        function.train(config, train_loader, model, criterion,
                       optimizer, epoch, writer_dict)

        # evaluate
        nme, predictions = function.validate(config, val_loader, model,
                                             criterion, epoch, writer_dict)

        is_best = nme < best_nme
        best_nme = min(nme, best_nme)

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        print("best:", is_best)
        utils.save_checkpoint(
            {"state_dict": model,
             "epoch": epoch + 1,
             "best_nme": best_nme,
             "optimizer": optimizer.state_dict(),
             }, predictions, is_best, final_output_dir, 'checkpoint_{}.pth'.format(epoch))

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()





#
# def main():
#
#     args = parse_args()
#
#     logger, final_output_dir, tb_log_dir = \
#         utils.create_logger(config, args.cfg, 'train')
#
#     logger.info(pprint.pformat(args))
#     logger.info(pprint.pformat(config))     # 打印参数到控制台和log文件
#
#     cudnn.benchmark = config.CUDNN.BENCHMARK    # 自动寻找最适合当前配置的高效算法
#     cudnn.determinstic = config.CUDNN.DETERMINISTIC     # 启用后，操作将使用可用的确定性算法
#     cudnn.enabled = config.CUDNN.ENABLED
#
#     model = models.get_face_alignment_net(config)   # 初始化一个模型，用yaml文件初始化config，用pretrained model初始化model
#
#     # copy model files
#     writer_dict = {
#         'writer': SummaryWriter(log_dir=tb_log_dir),
#         'train_global_steps': 0,
#         'valid_global_steps': 0,
#     }
#
#     gpus = list(config.GPUS)
#     model = nn.DataParallel(model, device_ids=gpus).cuda()
#
#     # loss
#     criterion = torch.nn.MSELoss(size_average=True).cuda()
#
#     optimizer = utils.get_optimizer(config, model)
#     best_nme = 100
#     last_epoch = config.TRAIN.BEGIN_EPOCH
#     if config.TRAIN.RESUME:
#         model_state_file = os.path.join(final_output_dir,
#                                         'latest.pth')
#         if os.path.islink(model_state_file):
#             checkpoint = torch.load(model_state_file)
#             last_epoch = checkpoint['epoch']
#             best_nme = checkpoint['best_nme']
#             model.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             print("=> loaded checkpoint (epoch {})"
#                   .format(checkpoint['epoch']))
#         else:
#             print("=> no checkpoint found")
#
#     if isinstance(config.TRAIN.LR_STEP, list):
#         lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
#             optimizer, config.TRAIN.LR_STEP,
#             config.TRAIN.LR_FACTOR, last_epoch-1
#         )
#     else:
#         lr_scheduler = torch.optim.lr_scheduler.StepLR(
#             optimizer, config.TRAIN.LR_STEP,
#             config.TRAIN.LR_FACTOR, last_epoch-1
#         )
#     dataset_type = get_dataset(config)
#
#     train_loader = DataLoader(
#         dataset=dataset_type(config,
#                              is_train=True),
#         batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
#         shuffle=config.TRAIN.SHUFFLE,
#         num_workers=config.WORKERS,
#         # num_workers=1,
#         pin_memory=config.PIN_MEMORY)
#
#     val_loader = DataLoader(
#         dataset=dataset_type(config,
#                              is_train=False),
#         batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
#         shuffle=False,
#         num_workers=config.WORKERS,
#         pin_memory=config.PIN_MEMORY
#     )
#
#     for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
#         lr_scheduler.step()
#
#         function.train(config, train_loader, model, criterion,
#                        optimizer, epoch, writer_dict)
#
#         # evaluate
#         nme, predictions = function.validate(config, val_loader, model,
#                                              criterion, epoch, writer_dict)
#
#         is_best = nme < best_nme
#         best_nme = min(nme, best_nme)
#
#         logger.info('=> saving checkpoint to {}'.format(final_output_dir))
#         print("best:", is_best)
#         utils.save_checkpoint(
#             {"state_dict": model,
#              "epoch": epoch + 1,
#              "best_nme": best_nme,
#              "optimizer": optimizer.state_dict(),
#              }, predictions, is_best, final_output_dir, 'checkpoint_{}.pth'.format(epoch))
#
#     final_model_state_file = os.path.join(final_output_dir,
#                                           'final_state.pth')
#     logger.info('saving final model state to {}'.format(
#         final_model_state_file))
#     torch.save(model.module.state_dict(), final_model_state_file)
#     writer_dict['writer'].close()
#
#
# if __name__ == '__main__':
#     main()





