# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import cv2
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel


class WFLW(data.Dataset):
    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP
        # 老子加的
        self.num_joints = cfg.MODEL.NUM_JOINTS  # 由于如果输入了wflw65就会改cfg中的关键点个数，此时这里就是65个节点，可以利用这个传参
        self.wflw65 = cfg.WFLW65  # 如果使用65节点的话会覆写这个 变成ture
        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])
        scale = self.landmarks_frame.iloc[idx, 1]

        center_w = self.landmarks_frame.iloc[idx, 2]
        center_h = self.landmarks_frame.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 4:].values  # 人工标记的关键点
        pts = pts.astype('float').reshape(-1, 2)
        scale *= 1.25
        # if (not self.is_train) and self.wflw65:  # 因为如果用65个节点 需要对其进行放大
        #     scale *= 1.25
        nparts = pts.shape[0]  # 获得点的个数    98个关键点
        face_brim_num = nparts - self.num_joints  # 0~32号关键点为人脸边缘关键点 i从33开始 否则就是从0开始
        # print(nparts)

        my_nparts = self.num_joints  # 65 关键点
        # print(my_nparts)

        # print(nparts.s)

        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        ###----------------------------------------------------------------###
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # print(img.stop)
        ###----------------------------------------------------------------###
        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,  # 随机缩放原图 增强鲁棒性(0.75~1.25)
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:  # 随机翻转图像在这里提现的
                img = np.fliplr(img)  # 实现图像左右反转
                pts = fliplr_joints(pts, width=img.shape[1], dataset='WFLW')  # 实现关键点坐高左右反转
                center[0] = img.shape[1] - center[0]
        ##
        # print("img.shape[1]:", img.shape[1])
        # print("center:", center.shape)
        ##
        # temp1 = [(pts[72][0] + pts[60][0]) / 2, (pts[72][1] + pts[60][1]) / 2]
        # temp2 = [(pts[82][0] + pts[76][0]) / 2, (pts[82][1] + pts[76][1]) / 2]
        # my_center = [(temp1[0] + temp2[0]) / 2, (temp1[1] + temp2[1]) / 2]
        # center[0] = my_center[0]  # 替换成我的centor
        # center[1] = my_center[1]
        ##
        # print("self.landmarks_frame.iloc[idx, 1]:", self.landmarks_frame.iloc[idx, 1])
        # print("scale:", scale)
        ##############################
        # if (not self.is_train) and self.wflw65:  # 因为如果用65个节点 需要对其进行放大
        if self.num_joints == 65: scale *= 0.85  # 自己设的 这个scale怀疑和训练有关，如果训练的时候没有设置这个，测试的时候设置，可能会对精度造成影响
        ##############################
        # print("scale:", scale)
        img = crop(img, center, scale, self.input_size, rot=r)
        ###----------------------------------------------------------------### 显示原图
        origin_img = np.uint8(img)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        # cv2.imshow("img", origin_img)   # origin_img (256, 256, 3) 才能显示原图 origin_img的格式是numpy.ndarray 数据类型为uint8
        # cv2.waitKey()                                               # 而pytorch能参与运算的格式是tensor，数据类型是float32
        # print(img.stop)
        ###----------------------------------------------------------------###  ↓ 高贵的原文
        # target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        # tpts = pts.copy()
        #
        # for i in range(nparts):
        #     if tpts[i, 1] > 0:  # tpts[i]是第i个点的坐标
        #         tpts[i, 0:2] = transform_pixel(tpts[i, 0:2] + 1, center,    # transform_pixel进行关键点的映射
        #                                        scale, self.output_size, rot=r)
        #         target[i] = generate_target(target[i], tpts[i] - 1, self.sigma,
        #                                     label_type=self.label_type)
        ###----------------------------------------------------------------###  ↑ 高贵的原文
        # 这里是我的部分，用于生成只剩下65个关键点的人脸关键点的映射，和target生成
        my_target = np.zeros((my_nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(my_nparts):  # 关键点个数改少了
            offset_i = i + face_brim_num
            if tpts[offset_i, 1] > 0:  # tpts[i]是第i个点的坐标 这里从第33个节点开始
                tpts[offset_i, 0:2] = transform_pixel(tpts[offset_i, 0:2] + 1, center,  # transform_pixel进行关键点的映射
                                                      scale, self.output_size, rot=r)
                my_target[i] = generate_target(my_target[i], tpts[offset_i] - 1, self.sigma,
                                               label_type=self.label_type)
        # cv2.imshow("img", origin_img)
        # cv2.waitKey()
        # print(img.stop)

        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        # target = torch.Tensor(target)
        # tpts = torch.Tensor(tpts)     # 直接把真实的关键点给改成65得了
        target = torch.Tensor(my_target)
        tpts = torch.Tensor(tpts[face_brim_num:])  # 都只取后面的
        pts = pts[face_brim_num:]  # 这个也是

        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}

        return img, target, meta, origin_img


if __name__ == '__main__':
    pass
