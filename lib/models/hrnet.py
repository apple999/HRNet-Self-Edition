# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Create by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):    # 残差网络中最基础的残差快，两个layer的那个
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):    # 残差网络中的第二种残差块，瓶颈设计的那种，瓶颈残差块
    expansion = 4   # 跟planes相乘用的......
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)     # 1*1*64*256
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,    # 3*3*64*64
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,  # 1*1*64*256
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,     # 正则
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):    # 先不添加新的分辨率流融合，先把已经存在的卷积流融合了
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(    # 此时为上斜线部分（上采样，获得上采样所用的卷积
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                    # nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:       # 此为下斜线部分（下采样，获得下采样所用的卷积
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))   # 第二阶段：[[None, (36,18,1,1,0)],[(18,36,3,2,1), None]]

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])   # 生成第二阶段的两种分辨率流的分别4个，共八个残差块，并且将上阶段x[]输入，生成输出x[]
            # 此时的x[]为没有聚合的阶段输出
        x_fuse = []
        for i in range(len(self.fuse_layers)):  # 这里太难看了.. 这里做的就是融合 如第二阶段的一个输出Y31=X21+fuse_layers[0][1](X22)
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])                           # 18  18    (36,18,1,1,0)   36
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:     # 上斜线部分（需要上采样部分
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),   # self.fuse_layers[0][1](x[1])
                        size=[x[i].shape[2], x[i].shape[3]],
                        mode='bilinear')    # 双线性插值法进行上采样
                else:   # 下斜线部分（下采样，卷积写好了，直接输入后加和就可以
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse   # 将该阶段的部分融合返回，此时的融合是不增加新分辨率stream的融合，如此时是阶段n，则返回n个融合好的输出，作为n+1阶段的前n个输入


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        self.inplanes = 64
        # self.cfg = config
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.sf = nn.Softmax(dim=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']  # [18,36]
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(     # 第一阶段和第二阶段衔接的那两个conv：
            [256], num_channels)                            # (256,18,3,1,1)这个只是通道变了;(256,36,3,2,1)这个不仅通道变了，分辨率还缩小了一倍
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']  # [18,36,72]
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)   # [18, 36], [18, 36, 72]
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        final_inp_channels = sum(pre_stage_channels)

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=final_inp_channels,
                kernel_size=1,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
            BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=config.MODEL.NUM_JOINTS,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

        # self.head = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=270,
        #         out_channels=270,
        #         kernel_size=1,
        #         stride=1,
        #         padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
        #     BatchNorm2d(270, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=270,
        #         out_channels=config.MODEL.NUM_JOINTS,
        #         kernel_size=extra.FINAL_CONV_KERNEL,
        #         stride=1,
        #         padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        # )

    def _make_transition_layer(     # 融合的第二部分，生成将下一个阶段新增加的输入
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)  # 2 第二阶段  # 3
        num_branches_pre = len(num_channels_pre_layer)  # 1         # 2

        transition_layers = []
        for i in range(num_branches_cur):   # 0, 1          # 0, 1, 2
            if i < num_branches_pre:        # 0<1           # 1<2
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:  # 256!=18
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],    # 256
                                  num_channels_cur_layer[i],    # 18
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:   # 这个主要是为了给增加的分辨率流准备的
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),  # 256, 36
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))  # 2to3阶段时 只有分辨率流2to3 [None,None,(36,72,3,2,1)]

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(     # 定义第一阶段的下采样
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):  # 范围[1, blocks)
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,  # 2   第二阶段            # 3
                                     block,         # 'BASIC'               # 'BASIC'
                                     num_blocks,    # [4, 4]                # [4, 4, 4]
                                     num_inchannels,    # [18, 36]          # [18, 36, 72]
                                     num_channels,      # [18, 36]          # [18, 36, 72]
                                     fuse_method,       # 'SUM'             # 'SUM
                                     reset_multi_scale_output)  # True      # True
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        # h, w = x.size(2), x.size(3)
        # print(self.cfg.MODEL.NUM_JOINTS)
        #
        # print("x.shape", x.shape)
        # print("x:", x)
        x = self.conv1(x)
        # print("x = self.conv1(x)")
        # print("x.shape", x.shape)
        x = self.bn1(x)
        # print("x = self.bn1(x)")
        # print("x.shape", x.shape)
        x = self.relu(x)
        # print("x = self.relu(x)")
        # print("x.shape", x.shape)
        x = self.conv2(x)
        # print("x = self.conv2(x)")
        # print("x.shape", x.shape)
        x = self.bn2(x)
        # print("x = self.bn2(x)")
        # print("x.shape", x.shape)
        x = self.relu(x)    # C/4,C/4,64 通道由3变为64，分辨率缩小4倍（作为主干网的输入）
        # print("x = self.relu(x)")
        # print("x.shape", x.shape)
        x = self.layer1(x)
        # print("x = self.layer1(x)")
        # print("x.shape", x.shape)
        # print(x.stop)
        # print("layer1->x.shape", x.shape)
        # self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        # self._make_layer返回一个layer[]里边有4个Bottleneck，每个输入输出都是64

        x_list = []     # 第二阶段的输入
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))   # 第一阶段进行到第四个残差块的x，进入transition[i]，生成下个阶段的两个输入x21,x22
            else:
                x_list.append(x)
        # print("阶段二输入：x_list[0].shape", x_list[0].shape)
        y_list = self.stage2(x_list)    # 跑的是HighResolutionModule(x_list)
        # print("阶段二输出：y_list[0].shape", y_list[0].shape)

        x_list = []     # 此时只有前两个分辨率的输入，第三个还没有
        for i in range(self.stage3_cfg['NUM_BRANCHES']):    # i = 0,1,2
            if self.transition2[i] is not None:     # self.transition2[] = [None,None,(36,72,3,2,1)]
                x_list.append(self.transition2[i](y_list[-1]))  # 对于新分辨率流的初始化（流3），只从上一个阶段的最后一个流（流2）下采样到新分辨率流，
            else:                                               # 并没有将上阶段的全部流融合（和论文里不一样
                x_list.append(y_list[i])
        # print("阶段三输入：x_list[0].shape", x_list[0].shape)
        y_list = self.stage3(x_list)    # x_list = [融合后的x31，融合后的x32，只有x22输入的x33]
        # print("阶段三输出：y_list[0].shape", y_list[0].shape)

        x_list = []     # 此时只有前三个分辨率的输入，第四个还没有
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:     # 生成新的分辨率流所需的卷积
                x_list.append(self.transition3[i](y_list[-1]))  # 进行卷积，生成新阶段所需要的第四个输入
            else:
                x_list.append(y_list[i])
        # print("阶段四输入：x_list[0].shape", x_list[0].shape)
        x = self.stage4(x_list)     # 融合
        # print("阶段四输出：x[0].shape", x[0].shape)

        # Head Part
        height, width = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(height, width), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x[2], size=(height, width), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x[3], size=(height, width), mode='bilinear', align_corners=False)
        # print("x[0]：x[0].shape", x[0].shape)
        # print("x1：x1.shape", x1.shape)
        # print("x2：x2.shape", x2.shape)
        # print("x3：x3.shape", x3.shape)
        x = torch.cat([x[0], x1, x2, x3], 1)
        # print("orix：x.shape", x.shape)
        x = self.head(x)

        # print("最终输出：x.shape", x.shape)

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():    # 返回该网络中的所有modules
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)    # 给tensor初始化，一般是给网络中参数weight初始化，初始化参数值符合正态分布。
                # nn.init.constant_(m.bias, 0)          # mean:均值，std:正态分布的标准差
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            # print("pretrained:", pretrained)
            pretrained_dict = torch.load(pretrained)    # Load all tensors onto the CPU
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)  # 将预训练模型的参数赋给model_dict
            self.load_state_dict(model_dict)


def get_face_alignment_net(config, **kwargs):

    model = HighResolutionNet(config, **kwargs)     # 初始化model 为HighResolutionNet类
    pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
    # print("config.MODEL.PRETRAINED:", config.MODEL.PRETRAINED)
    # print("pretrained:", pretrained)
    model.init_weights(pretrained=pretrained)   # 将pretrained model值给model

    return model

