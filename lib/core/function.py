# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import cv2
import torch
import numpy as np

from .evaluation import decode_preds, compute_nme
from pathlib import Path

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    for i, (inp, target, meta, _) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        # compute the output
        # print("inp:", inp)
        # print("inp.size:", inp.size)

        # print("inp.shape:", inp.shape)

        output = model(inp)
        # print("output:", output[0][0])
        # print("output.shape:", output.shape)

        # print("target:", target)
        # print("target.shape:", target.shape)
        target = target.cuda(non_blocking=True)  # target代表的是标签值
        # print("target2:", target)
        # print("target2.shape:", target.shape)

        loss = critertion(output, target)
        # print("----loss----:", loss)
        # print("loss.shape:", loss.shape)

        # if i == 10:
        #
        #     print(inp.stophere)

        # NME
        score_map = output.data.cpu()
        preds, preds_x64 = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

        nme_batch = compute_nme(preds, meta)
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time() - end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=inp.size(0) / batch_time.val,
                data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}' \
        .format(epoch, batch_time.avg, losses.avg, nme)
    logger.info(msg)


def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta, _) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)

            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds, preds_x64 = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # NME
            nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions


def inference(config, data_loader, model, draw_flag, root_output_dir, evel65=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()
    ##
    mse_lefteye_count = 0
    mse_righteye_count = 0
    mse_nose_batch_count = 0
    mse_mouth_batch_count = 0
    ##
    mse_lefteye_batch_sum = 0
    mse_righteye_batch_sum = 0
    mse_nose_batch_sum = 0
    mse_mouth_batch_sum = 0
    ##
    cnt = 0
    if draw_flag:
        # root_output_dir = './output_withpred'
        dataset = config.DATASET.DATASET
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        final_output_dir = root_output_dir + '/' + dataset + '/' + time_str
        print('=> creating {}'.format(final_output_dir))
        Path(final_output_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, (inp, target, meta, origin_img) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)

            ###-------------------------------------###
            origin_img = origin_img.numpy()
            # for j in range(len(inp)):
            #     print("origin_img:", origin_img.shape)  # origin_img (256, 256, 3) uint8
                # print("origin_img:", origin_img[j])  # origin_img (256, 256, 3) uint8
                # cv2.imshow("origin_img[{}]".format(j), origin_img[j])   # 至此，可以输出原始图像
                # cv2.waitKey()
                # print(j.stop)

            ###-------------------------------------###

            score_map = output.data.cpu()  # score_map 和 output 是同一个东西 data.cpu() 是把output从cuda（gpu）转换到了cpu

            preds, preds_x64 = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])  # 预测点在原图的坐标（8, 98, 2）
            # meta['tpts'] 人工标定点在64*64的坐标（8,98,2）
            if draw_flag:
                out_img = origin_img
                for j in range(len(inp)):
                    pic_num = '{:0>8}'.format(cnt)
                    for k in range(len(meta['tpts'][j])):
                        dot = meta['tpts'][j][k]
                        pred_dot = preds_x64[j][k]
                        cv2.circle(out_img[j], (int(dot.tolist()[0] * 4.0), int(dot.tolist()[1] * 4.0)), 1, (0, 0, 255),
                                   -1)  # 进行绘图
                        cv2.circle(out_img[j], (int(pred_dot.tolist()[0] * 4.0), int(pred_dot.tolist()[1] * 4.0)), 1,
                                   (0, 255, 0), -1)  # 进行绘图
                    # cv2.imshow("out_img", out_img[j])
                    # cv2.waitKey()
                    # print(dot.s)

                    final_pic_path = final_output_dir + '/' + pic_num + '.jpg'
                    print('=> creating {}'.format(final_pic_path))
                    cv2.imwrite(final_pic_path, out_img[j])
                    cnt += 1
            else:
                for j in range(len(inp)):   # 测试结果可视化
                    pic_num = '{:0>8}'.format(cnt)
                    print('=> testing {}'.format(pic_num+'.jpg'))
                    cnt += 1

            # NME
            nme_temp, mse_lefteye, mse_righteye, mse_nose, mse_mouth = compute_nme(preds, meta, evel65, cnt)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            if failure_008 < 0.00007: print("{}~{}".format(cnt-8, cnt))
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            ## 加的算每个地方的nme
            mse_lefteye_batch_sum += np.sum(mse_lefteye)
            mse_righteye_batch_sum += np.sum(mse_righteye)
            mse_nose_batch_sum += np.sum(mse_nose)
            mse_mouth_batch_sum += np.sum(mse_mouth)
            ##
            nme_count = nme_count + preds.size(0)
            ##
            mse_lefteye_count = mse_lefteye_count + 18
            mse_righteye_count = mse_righteye_count + 18
            mse_nose_batch_count = mse_nose_batch_count + 9
            mse_mouth_batch_count = mse_mouth_batch_count + 20
            ##
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    nme_lefteye = mse_lefteye_batch_sum / mse_lefteye_count
    nme_righteye = mse_righteye_batch_sum / mse_righteye_count
    nme_nose = mse_nose_batch_sum / mse_nose_batch_count
    nme_mouth = mse_mouth_batch_sum / mse_mouth_batch_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)
    msg2 = 'nme_lefteye:{:.4f}, nme_righteye:{:.4f}, nme_nose:{:.4f}, nme_mouth:{:.4f}'.format(nme_lefteye, nme_righteye, nme_nose, nme_mouth)
    logger.info(msg2)

    return nme, predictions
