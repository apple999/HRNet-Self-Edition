# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math

import torch
import numpy as np

from ..utils.transforms import transform_preds


def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def compute_nme(preds, meta, evel65=False, cnt=-1):
    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()
    offset = 0
    N = preds.shape[0]
    L = preds.shape[1]
    if evel65 and L == 98:
        offset = 33
    # print(preds.shape)
    print("N:{}, L:{}, offset:{}, evel65:{}".format(N, L, offset, evel65))
    # for i in range(N):
    #     print(i)
    # print(N.ss)
    rmse = np.zeros(N)
    ### 添加的针对各个地方的精度计算
    rmse_lefteye = np.zeros(N)
    rmse_righteye = np.zeros(N)
    rmse_nose = np.zeros(N)
    rmse_mouth = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        # print("preds.shape:{}, target.shape:{}".format(preds.shape, target.shape))
        # print("pts_pred.shape:{}, pts_gt.shape:{}".format(pts_pred.shape, pts_gt.shape))
        # print(N.ss)
        if evel65 and L == 98:  # 如果是98个节点的模型进行测试，为保证公平，只验证其中局部关键点的准确度
            pts_pred = pts_pred[33:, :]
            pts_gt = pts_gt[33:, :]
        # print(evel65 and L == 98)
        # print("pts_pred.shape:{}, pts_gt.shape:{}".format(pts_pred.shape, pts_gt.shape))
        # 98：
        # [33,41] 左眉毛 [60,67]+96 左眼睛
        # [42,50] 右眼睛 [68,75]+97 右眼睛
        # [51,59] 鼻子
        # [76,95] 嘴
        # 65：
        # [0, 8] 左眉毛  [27, 34] + 63 左眼睛
        # [9, 17] 右眼睛  [35, 42] + 64 右眼睛
        # [18, 26] 鼻子
        # [43, 62] 嘴

        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:  # wflw
            interocular = np.linalg.norm(pts_gt[60 - offset, ] - pts_gt[72 - offset, ])
        elif L == 65:  # wflw-65
            interocular = np.linalg.norm(pts_gt[27, ] - pts_gt[39, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        # if evel65 & L == 98:
        #     rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * (L-offset))
        # else:
        #     rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * (L - offset))
        ### 添加的针对各个地方的精度计算
        pred_left_eye = np.concatenate((pts_pred[0:9, :], pts_pred[27:35, :], pts_pred[63, :].reshape(1, 2)), axis=0)
        gt_left_eye = np.concatenate((pts_gt[0:9, :], pts_gt[27:35, :], pts_gt[63, :].reshape(1, 2)), axis=0)
        pred_right_eye = np.concatenate((pts_pred[9:18, :], pts_pred[35:43, :], pts_pred[64, :].reshape(1, 2)), axis=0)
        gt_right_eye = np.concatenate((pts_gt[9:18, :], pts_gt[35:43, :], pts_gt[64, :].reshape(1, 2)), axis=0)
        pred_nose = pts_pred[18:27, :]
        gt_nose = pts_gt[18:27, :]
        pred_mouth = pts_pred[43:63, :]
        gt_mouth = pts_gt[43:63, :]
        #
        rmse_lefteye[i] = np.sum(np.linalg.norm(pred_left_eye - gt_left_eye, axis=1)) / (interocular * (pred_left_eye.shape[0]))
        rmse_righteye[i] = np.sum(np.linalg.norm(pred_right_eye - gt_right_eye, axis=1)) / (interocular * (pred_right_eye.shape[0]))
        rmse_nose[i] = np.sum(np.linalg.norm(pred_nose - gt_nose, axis=1)) / (interocular * (pred_nose.shape[0]))
        rmse_mouth[i] = np.sum(np.linalg.norm(pred_mouth - gt_mouth, axis=1)) / (interocular * (pred_mouth.shape[0]))
        # if rmse[i] <= 0.10 and cnt != -1: print("{}的测试nme为{}，小于0.10".format(cnt, rmse[i]))
        # 这里是为了计算各个部分的均方差而设计的，训练不可用 会报错，训练时需要改
    return rmse, rmse_lefteye, rmse_righteye, rmse_nose, rmse_mouth


def decode_preds(output, center, scale, res):
    coords = get_preds(output)  # float type

    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()
    # print("preds in evaluation.py:", preds)     # 算出来的64*64的关键点图
    preds_x64 = preds.clone()
    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())
    # print("preds222222 in evaluation.py:", preds)   # 转换到原图的的关键点图
    return preds, preds_x64
