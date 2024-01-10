#!/usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%


import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch

# 检查opencv的版本
# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3:  # pragma: no cover
    print('Warning: OpenCV 3 is not installed')

# 颜色映射算法Jet，可视化
# Jet colormap for visualization.
myjet = np.array([[0., 0., 0.5],
                  [0., 0., 0.99910873],
                  [0., 0.37843137, 1.],
                  [0., 0.83333333, 1.],
                  [0.30044276, 1., 0.66729918],
                  [0.66729918, 1., 0.30044276],
                  [1., 0.90123457, 0.],
                  [1., 0.48002905, 0.],
                  [0.99910873, 0.07334786, 0.],
                  [0.5, 0., 0.]])

# 主要部分
# 网络模型
class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    # 定义一些网络层
    def __init__(self):
        super(SuperPointNet, self).__init__()
        # 定义relu函数，非线性激活函数
        self.relu = torch.nn.ReLU(inplace=True)
        # 定义maxpooling层，2x2的maxpooling
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义不同的通道数，
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256

        """ torch.nn.Conv2d(输入通道，输出通道，卷积核大小，步长，padding)"""
        """
                                    -> Interest Point Decoder  -> 特征点位置
        input  -->  Shared Encoder <
                                    -> Descriptor Decoder      -> 描述子
+-----------------+----------------+-----------+------------+-------------+--------+--------+---------------+
|      module     |   layer_name   | inputsize | in_channel | out_channel | kernel | stride |   outputsize  |
+-----------------+----------------+-----------+------------+-------------+--------+--------+---------------+
|      Input      |                |    H*W    |     1      |             |        |        |               |
+-----------------+----------------+-----------+------------+-------------+--------+--------+---------------+
|     Encoder     |  conv1a+Relu   |    H*W    |     1      |      64     |  3*3   |   1    |      H*W      |
|                 |  conv1b+Relu   |    H*W    |     64     |      64     |  3*3   |   1    |      H*W      |
|                 |  max-pooling1  |    H*W    |            |             |  2*2   |        |    H/2*W/2    |
|                 |  conv2a+Relu   |  H/2*W/2  |     64     |      64     |  3*3   |   1    |    H/2*W/2    |
|                 |  conv2b+Relu   |  H/2*W/2  |     64     |      64     |  3*3   |   1    |    H/2*W/2    |
|                 |  max-pooling2  |  H/2*W/2  |            |             |  2*2   |        |    H/4*W/4    |
|                 |  conv3a+Relu   |  H/4*W/4  |     64     |     128     |  3*3   |   1    |    H/4*W/4    |
|                 |  conv3b+Relu   |  H/4*W/4  |    128     |     128     |  3*3   |   1    |    H/4*W/4    |
|                 |  max-pooling3  |  H/4*W/4  |            |             |  2*2   |        |    H/8*W/8    |
|                 |  conv4a+Relu   |  H/8*W/8  |    128     |     128     |  3*3   |   1    |    H/8*W/8    |
|                 |  conv4b+Relu   |  H/8*W/8  |    128     |     128     |  3*3   |   1    |    H/8*W/8    |
+-----------------+----------------+-----------+------------+-------------+--------+--------+---------------+
| Interest Point  |     convPa     |  H/8*W/8  |    128     |     256     |  3*3   |   1    |    H/8*W/8    |
|     Decoder     |     convPb     |  H/8*W/8  |    256     |      65     |  1*1   |        |  [H/8,W/8,65] |
|                 |    Softmax     |           |            |             |        |        |  [H/8,W/8,64] |
|                 |    Reshape     |           |            |             |        |        |    [H,W,1]    |
+-----------------+----------------+-----------+------------+-------------+--------+--------+---------------+
|   Descriptor    |     convDa     |  H/8*W/8  |    128     |     256     |  3*3   |   1    |    H/8*W/8    |
|     Decoder     |     convDb     |  H/8*W/8  |    256     |     256     |  1*1   |   1    | [H/8,W/8,256] |
|                 | Bi-Interpolate |           |            |             |        |        |   [H,W,256]   |
|                 |    L2-Norm     |           |            |             |        |        |   [H,W,256]   |
+-----------------+----------------+-----------+------------+-------------+--------+--------+---------------+

    """
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
        # 对于assets/icl_snippet，H = 120, W=160
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        """
    desc.shape = [1, 256, H/8, W/8]
    torch.norm表示对desc的dimension=1的数据，求2范数
    torch.unsqueeze对dn的维度=1插入一个维度；dn维度变化：[1,H/8,W/8] --> [1,1,H/8,W/8]
    div除法，让desc的维度1上的256通道的数据除以dn，完成归一化
    """

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        #print(dn.shape)
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        """
    最终输出:
    desc.shape = [1, 256, H/8, W/8]，这个是Descriptor Decoder中网络的最后输出
    semi.shape = [1, 65 , H/8, W/8]，这个是Interest Point Decoder中网络的最后输出
    """

        return semi, desc


class SuperPointFrontend(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
                 cuda=False):
        self.name = 'SuperPoint'  # 名称
        self.cuda = cuda  # 是否使用cuda
        self.nms_dist = nms_dist  # nms的半径距离
        self.conf_thresh = conf_thresh  # 筛选置信值的阈值
        self.nn_thresh = nn_thresh  # 最近邻匹配的距离阈值，没用上           L2 descriptor distance for good match.
        self.cell = 8  # 每个cell里面是8x8，跟论文一致  Size of each output cell. Keep this fixed.
        self.border_remove = 4  # 去除边界的点                 Remove points this close to the border.

        # Load the network in inference mode.
        # 加载模型和权重，并且判断是否在cuda上运行
        self.net = SuperPointNet()
        if cuda:
            # Train on GPU, deploy on GPU.
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(torch.load(weights_path,
                                                map_location=lambda storage, loc: storage))
        # 测试模型，运行
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
    对于输入的形状为 3xN [x_i,y_i,conf_i]^T的numpy类型的特征点数组，进行一个快速近似的NMS。输入形状如下
          +-----------+--------+------------+-----------------+
          |    x_0    |  x_1   |    ...     |      x_N-1      |
          |    y_0    |  y_1   |    ...     |      y_N-1      |
          |   conf_0  | conf_1 |    ...     |     conf_N-1    |
          +-----------+--------+------------+-----------------+
    算法总结：
      创建一个大小为HxW的网格，让每个特征点的位置为1，其他的为0.
      遍历所有等于1的位置，把它们变成-1或者0. -1表示保留，0表示抑制

      本质上：<<<           通过将置信值最大点附近的值设置为0来抑制        >>>
    """
        """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        """
    1.根据置信值进行排序，这里先取相反数再从小到大排序，最终返回的是由大到小的索引
    2.同时进行对x_i和y_i取整
        argsort 用来将列表中的元素进行从小到大排列，返回的是一串索引
    3.检查边界条件，只有0个或者1个点的情况。直接返回0个或者1个特征点
    """
        #print(in_corners.shape)
        inds1 = np.argsort(-in_corners[2, :])  # 原始输入的置信值从高到低排序
        corners = in_corners[:, inds1]  # [3,N]，按照置信值从高到底排序[x,y,conf]

        # 这里仅对x_i和y_i操作，注意切片操作
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners. [2,N]

        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            # np.vstack 按照行方向叠加；[rcorners[0],rcorners[1],in_corners]^T
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        """
    1. 初始化网格grid，把特征点位置填上1，同时记录索引
    2. 对边界进行padding，方便对边界点也进行nms，padding的距离就是NMS的半径距离，dist_thresh
    这里注意从特征点坐标，到grid映射的关系
    +----------+---------+---------+---------+---------+
                               W
      |    *     |    ——   |    ——   |    >    | x轴方向 |
      |    |     |    *    |    *    |    *    |    *    |
    H |    |     |    *    |    *    |    *    |    *    |
      |    v     |    *    |    *    |    *    |    *    |
      | y轴方向   |    *    |    *    |    *    |    *    |
    +----------+---------+---------+---------+---------+
    """
        # 1.Initialize the grid.
        # 这些有点迷惑性，其实没有用到rcorners.T，还是对[2,N]进行遍历，把特征点位置填上1，同时记录索引
        for i, rc in enumerate(rcorners.T):
            # 这里映射坐标是[y,x]
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # 2.Pad the border of the grid, so that we can NMS points near the border.
        """
                                [[0 0 0 0 0]
         [[1 1 1]      --->      [0 1 1 1 0]
          [1 1 1]]     --->      [0 1 1 1 0]
                                 [0 0 0 0 0]]
    """
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')  # constant表示用同一个常数填充，默认是0

        """
    开始遍历所有点，根据置信值从高到低开始抑制，rcorners.T=[N,2] 
      padding后的坐标变换，[x,y]  -->  [x+pad , y+pad],  pad = dist_thresh
      如果网格(grid)里面这个点值是1，表示还没有处理（-1表示保留，0表示被抑制，1表示还未处理）
        对于[pt_x,pt_y]周围半径为pad的区域全部设置为0，再把[pt_x,pt_y]位置设置为-1，表示抑制周围所有点
        +---------------------+---------+-------------+---------+---------------------+
        | [pt_x-pad,pt_y-pad] |    *    |      *      |    *    | [pt_x-pad,pt_y+pad] |
        |          *          |    *    |      *      |    *    |          *          |
        |   [pt_x,pt_y-pad]   |    *    | [pt_x,pt_y] |    *    |   [pt_x,pt_y+pad]   |
        |          *          |    *    |      *      |    *    |          *          |
        | [pt_x+pad,pt_y-pad] |    *    |      *      |    *    | [pt_x+pad,pt_y+pad] |
        +---------------------+---------+-------------+---------+---------------------+

    """
        #这里有个小思考，为什么要按照置信值从高到低开始抑制？有什么好处？
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0  # NMS之后，真正的特征点个数
        for i, rc in enumerate(rcorners.T):
            # rc=[x,y],考虑pad后的坐标转换，比如原来是(0,0)，加入padding之后，就变成(pad,pad)
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)  # 返回grid中值为-1的位置，根据开始映射关系，返回是[y,x]
        keepy, keepx = keepy - pad, keepx - pad  # 移除pad的坐标变换
        inds_keep = inds[keepy, keepx]  # 返回保留特征点对应的索引
        out = corners[:, inds_keep]  # 按照索引得到所需特征点，corners:按照置信值从高到底排序[x,y,conf]
        values = out[-1, :]  # 对应置信值
        inds2 = np.argsort(-values)  # 重新按照置信值排序，从高到低
        out = out[:, inds2]  # 将输出特征点按照置信值从高到低排序
        # inds1:原始输入按置信值从高到低排序，原始的索引,
        # inds_keep:返回保留特征点对应的inds_keep索引,
        # inds2:保留特征点的置信值从高到低排序inds_keep索引
        out_inds = inds1[inds_keep[inds2]]  # 输出out中点对应的原始索引，其实后面没用，后面有更直接获取的方法
        return out, out_inds

    def run(self, img):
        """ Process a numpy image to extract points and descriptors.
    Input
      img - HxW numpy float32 input image in range [0,1].
    Output
      corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      desc - 256xN numpy array of corresponding unit normalized descriptors.
      heatmap - HxW numpy heatmap in range [0,1] of point confidences.
      """
        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[0], img.shape[1]  # 获取输入图片维度
        inp = img.copy()
        inp = (inp.reshape(1, H, W))
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(1, 1, H, W)  # 自动求导
        if self.cuda:
            inp = inp.cuda()
        """
    1.把图像输入到网络中，拿到输出
        semi       ：这个是Interest Point Decoder中网络的最后输出
        coarse_desc：这个是Descriptor Decoder中网络的最后输出
    2.Interest Point Decoder后处理
      2.1：对65通道执行softmax
      2.2：丢弃dustbin，65 -> 64
      2.3：reshape， H/8*W/8  ->  H*W
      2.4：去除置信值小于阈值的点
      2.5：进行NMS
      2.6：按照置信值从低到高排序
      2.7：去掉边界附近的点
      2.8：得到最终的特征点
    3.Descriptor Decoder后处理
      3.1：插值
    """
        # Forward pass of network.
        outs = self.net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        # Convert pytorch -> numpy.同时减少了一个维度，[1, 65 , H/8, W/8] -> [65 , H/8, W/8]
        semi = semi.data.cpu().numpy().squeeze()
        # 2.Interest Point Decoder后处理
        # --- Process points.
        # 2.1 softmax
        dense = np.exp(semi)  # Softmax.
        dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
        # 2.2 Remove dustbin.
        nodust = dense[:-1, :, :]
        # 2.3 Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)  # H/8
        Wc = int(W / self.cell)  # W/8
        nodust = nodust.transpose(1, 2, 0)  # [64 , H/8, W/8]    -> [H/8, W/8, 64]
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])  # [H/8, W/8, 64]     -> [H/8, W/8, 8, 8]
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])  # [H/8, W/8, 8, 8]   -> [H/8, 8  , W/8, 8]
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])  # [H/8, 8  , W/8, 8] -> [H , W]
        # 2.4 去除置信值小于阈值的点
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        # 2.5 NMS
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        # 2.6 按照置信值从低到高排序
        inds = np.argsort(pts[2, :])  # 按照NMS之后点的置信值从低到高排序的索引
        pts = pts[:, inds[::-1]]  # Sort by confidence.将点按照置信值从低到高的顺序排列
        # 2.7 Remove points along border.去除边界附近的点
        """
    ---------------------
    ---***************---
    ---***************---
    ---***************---
    ---------------------
    """
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        # 2.8 得到最终的特征点
        pts = pts[:, ~toremove]  ########################最终的特征点##########################
        # 3.Descriptor Decoder后处理
        # --- Process descriptor.
        D = coarse_desc.shape[1]  # coarse_desc=[1, 256, H/8, W/8],所以D=256
        if pts.shape[1] == 0:  # 没有特征点
            desc = np.zeros((D, 0))
        else:
            # 3.1 插值
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())  # 拿到[x,y]坐标，[2,N]
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.  #
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()#[N,2]
            samp_pts = samp_pts.view(1, 1, -1, 2)#[1,1,N,2]
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.cuda()
            # 默认是bilinear(双线性插值)
            """
            torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
            input.shape=(N,C,Hin,Win)
            grid.shape =(N,Hout,Wout,2), grid的范围是[-1,1], [-1,-1]表示左上角的点
            output.shape=(N,C,Hout,Wout)
            在这里输入是[1,256,H/8,W/8]，grid是[1,1,N,2]；所以输出是[1,256,1,N]
            align_corners=true, (-1,-1)表示在像素块中心，反之在像素块左上角
            """
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)  # [256,N]
            # 标准化，先对256通道方向求2范数，再逐通道除以范数，完成标准化
            # np.linalg.norm表示对维度0求2范数，np.newaxis表示插入新维度
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
            """
      pts:[2,N],特征点坐标,[x,y]
      desc:[256,N]，对应描述子
      heatmap:[3,N]，特征点坐标和置信值，[x,y,conf]
      """
        return pts, desc, heatmap


# 下面主要是绘图类相关的，没有深入阅读，简要注释一下
class PointTracker(object):
    """ Class to manage a fixed memory of points and descriptors that enables
  sparse optical flow point tracking.

  Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
  tracks with maximum length L, where each row corresponds to:
  row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
  """

    def __init__(self, max_length, nn_thresh):
        if max_length < 2:
            raise ValueError('max_length must be greater than or equal to 2.')
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = []
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl + 2))
        self.track_count = 0
        self.max_score = 9999
    #mutual check
    def nn_match_two_way(self, desc1, desc2, nn_thresh):
        """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        if nn_thresh < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')
        # Compute L2 distance. Easy since vectors are unit normalized.计算距离矩阵
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        # Check if nearest neighbor goes both directions and keep those.交叉检验
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.获取最终的匹配索引
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        # Populate the final 3xN match data structure.
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores
        return matches

    def get_offsets(self):
        """ Iterate through list of points and accumulate an offset value. Used to
    index the global point IDs into the list of points.

    Returns
      offsets - N length array with integer offset locations.
    """
        # Compute id offsets.
        offsets = []
        offsets.append(0)
        for i in range(len(self.all_pts) - 1):  # Skip last camera size, not needed.
            offsets.append(self.all_pts[i].shape[1])
        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)  # 当成1维数组进行累加和，1，1+2，1+2+3,...
        return offsets

    def update(self, pts, desc):
        """ Add a new set of point and descriptor observations to the tracker.

    Inputs
      pts - 3xN numpy array of 2D point observations.
      desc - DxN numpy array of corresponding D dimensional descriptors.
    """
        if pts is None or desc is None:
            print('PointTracker: Warning, no points were added to tracker.')
            return
        assert pts.shape[1] == desc.shape[1]
        # Initialize last_desc.
        if self.last_desc is None:
            self.last_desc = np.zeros((desc.shape[0], 0))
        # Remove oldest points, store its size to update ids later.
        remove_size = self.all_pts[0].shape[1]  # 上一帧的特征点个数
        self.all_pts.pop(0)
        self.all_pts.append(pts)
        # Remove oldest point in track.
        self.tracks = np.delete(self.tracks, 2, axis=1)  # 删除第2列，最老的特征点索引
        # Update track offsets.
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        offsets = self.get_offsets()
        # Add a new -1 column.
        self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))
        # Try to append to existing tracks.
        matched = np.zeros((pts.shape[1])).astype(bool)
        matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
        for match in matches.T:
            """
      存放全局的索引：
      [[-------frame1--------]，[-------frame2--------]，[-------frame3--------]，[-------frame4--------]]
      把相对索引换成全局索引：
      [[-------frame1--------]，[-------frame2--------]，[1st                  ]，[ 2nd                 ]]
      """
            # Add a new point to it's matched track.
            id1 = int(match[0]) + offsets[-2]  # 上一帧的点索引，因为match里面是相对于某一帧的索引，但是后面track要的是全局的点索引
            id2 = int(match[1]) + offsets[-1]  # 最新的点索引
            found = np.argwhere(self.tracks[:, -2] == id1)
            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2
                if self.tracks[row, 1] == self.max_score:
                    # Initialize track score.
                    self.tracks[row, 1] = match[2]
                else:
                    # Update track score with running average.
                    # NOTE(dd): this running average can contain scores from old matches
                    #           not contained in last max_length track points.
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.
                    frac = 1. / float(track_len)
                    self.tracks[row, 1] = (1. - frac) * self.tracks[row, 1] + frac * match[2]
        # Add unmatched tracks.
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]
        new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_num = new_ids.shape[0]
        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num  # Update the track count.
        # Remove empty tracks.
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]
        # Store the last descriptors.
        self.last_desc = desc.copy()
        return

    def get_tracks(self, min_length):
        """ Retrieve point tracks of a given minimum length.
    Input
      min_length - integer >= 1 with minimum track length
    Output
      returned_tracks - M x (2+L) sized matrix storing track indices, where
        M is the number of tracks and L is the maximum track length.
    """
        if min_length < 1:
            raise ValueError('\'min_length\' too small.')
        valid = np.ones((self.tracks.shape[0])).astype(bool)
        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        # Remove tracks which do not have an observation in most recent frame.
        not_headless = (self.tracks[:, -1] != -1)
        keepers = np.logical_and.reduce((valid, good_len, not_headless))
        returned_tracks = self.tracks[keepers, :].copy()
        return returned_tracks

    def draw_tracks(self, out, tracks):
        """ Visualize tracks all overlayed on a single image.
    Inputs
      out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
      tracks - M x (2+L) sized matrix storing track info.
    """
        # Store the number of points per camera.
        pts_mem = self.all_pts
        N = len(pts_mem)  # Number of cameras/images.
        # Get offset ids needed to reference into pts_mem.
        offsets = self.get_offsets()
        # Width of track and point circles to be drawn.
        stroke = 1
        # Iterate through each track and draw it.
        for track in tracks:
            clr = myjet[int(np.clip(np.floor(track[1] * 10), 0, 9)), :] * 255
            for i in range(N - 1):
                if track[i + 2] == -1 or track[i + 3] == -1:
                    continue
                offset1 = offsets[i]
                offset2 = offsets[i + 1]
                idx1 = int(track[i + 2] - offset1)
                idx2 = int(track[i + 3] - offset2)
                pt1 = pts_mem[i][:2, idx1]
                pt2 = pts_mem[i + 1][:2, idx2]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                # 绘制线
                cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
                # Draw end points of each track.
                if i == N - 2:
                    clr2 = (255, 0, 0)
                    cv2.circle(out, p2, stroke, clr2, -1, lineType=16)


"""
处理输入数据，支持三种类型
1. usb 摄像头
2. 放有图片的文件夹
3. 视频，切割成图片集，转成跟2类似的情况
注意一下：read_image函数
        输入的图片是灰度图并且做了归一化的，如果不这样，直接输入到网络，差别还是挺大的(基于个人实验的结果，如有不同正常)
"""


class VideoStreamer(object):
    """ Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """

    def __init__(self, basedir, camid, height, width, skip, img_glob):
        self.cap = []
        self.camera = False
        self.video_file = False
        self.listing = []
        self.sizer = [height, width]
        self.i = 0
        self.skip = skip
        self.maxlen = 1000000
        # If the "basedir" string is the word camera, then use a webcam.
        if basedir == "camera/" or basedir == "camera":
            print('==> Processing Webcam Input.')
            self.cap = cv2.VideoCapture(camid)
            self.listing = range(0, self.maxlen)
            self.camera = True
        else:
            # Try to open as a video.
            self.cap = cv2.VideoCapture(basedir)
            lastbit = basedir[-4:len(basedir)]
            if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
                raise IOError('Cannot open movie file')
            elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
                print('==> Processing Video Input.')
                num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.listing = range(0, num_frames)
                self.listing = self.listing[::self.skip]
                self.camera = True
                self.video_file = True
                self.maxlen = len(self.listing)
            else:
                print('==> Processing Image Directory Input.')
                search = os.path.join(basedir, img_glob)
                self.listing = glob.glob(search)
                self.listing.sort()
                self.listing = self.listing[::self.skip]
                self.maxlen = len(self.listing)
                if self.maxlen == 0:
                    raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

    def read_image(self, impath, img_size):
        """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        # Image is resized via opencv.
        interp = cv2.INTER_AREA
        grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
        grayim = (grayim.astype('float32') / 255.)
        return grayim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
        """
    如果是摄像头，就实时获取
    如果不是，则从已经加载好的图像集合里面加载,i+1
    """
        if self.i == self.maxlen:
            return (None, False)
        if self.camera:
            ret, input_image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
                return (None, False)
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
            input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                                     interpolation=cv2.INTER_AREA)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            input_image = input_image.astype('float') / 255.0
        else:
            image_file = self.listing[self.i]
            input_image = self.read_image(image_file, self.sizer)
        # Increment internal counter.
        self.i = self.i + 1
        input_image = input_image.astype('float32')
        return (input_image, True)


if __name__ == '__main__':

    # Parse command line arguments.可以参考github或者看注释都行
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory or movie file or "camera" (for webcam).')
    parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
                        help='Path to pretrained weights file (default: superpoint_v1.pth).')
    parser.add_argument('--img_glob', type=str, default='*.png',
                        help='Glob match if directory of images is specified (default: \'*.png\').')
    parser.add_argument('--skip', type=int, default=1,
                        help='Images to skip if input is movie or directory (default: 1).')
    parser.add_argument('--show_extra', action='store_true',
                        help='Show extra debug outputs (default: False).')
    parser.add_argument('--H', type=int, default=120,
                        help='Input image height (default: 120).')
    parser.add_argument('--W', type=int, default=160,
                        help='Input image width (default:160).')
    parser.add_argument('--display_scale', type=int, default=2,
                        help='Factor to scale output visualization (default: 2).')
    parser.add_argument('--min_length', type=int, default=2,
                        help='Minimum length of point tracks (default: 2).')
    parser.add_argument('--max_length', type=int, default=5,
                        help='Maximum length of point tracks (default: 5).')
    parser.add_argument('--nms_dist', type=int, default=4,
                        help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
                        help='Detector confidence threshold (default: 0.015).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
                        help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--camid', type=int, default=0,
                        help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
    parser.add_argument('--waitkey', type=int, default=1,
                        help='OpenCV waitkey time in ms (default: 1).')
    parser.add_argument('--cuda', action='store_true',
                        help='Use cuda GPU to speed up network processing speed (default: False)')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images to screen. Useful if running remotely (default: False).')
    parser.add_argument('--write', action='store_true',
                        help='Save output frames to a directory (default: False)')
    parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
                        help='Directory where to write output frames (default: tracker_outputs/).')
    opt = parser.parse_args()
    print(opt)

    """
  1.加载输入对象
  2.加载模型
  3.初始化
  4.开始运行demo
    4.1 获取新的一帧
    4.2 提点
    4.3 track
    4.4 绘图
  5.结束
  """

    # This class helps load input images from different sources.
    vs = VideoStreamer(opt.input, opt.camid, opt.H, opt.W, opt.skip, opt.img_glob)

    print('==> Loading pre-trained network.')
    # This class runs the SuperPoint network and processes its outputs.
    fe = SuperPointFrontend(weights_path=opt.weights_path,
                            nms_dist=opt.nms_dist,
                            conf_thresh=opt.conf_thresh,
                            nn_thresh=opt.nn_thresh,
                            cuda=opt.cuda)
    print('==> Successfully loaded pre-trained network.')

    # This class helps merge consecutive point matches into tracks.
    tracker = PointTracker(opt.max_length, nn_thresh=fe.nn_thresh)

    # Create a window to display the demo.
    if not opt.no_display:
        win = 'SuperPoint Tracker'
        cv2.namedWindow(win)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Font parameters for visualizaton.
    font = cv2.FONT_HERSHEY_DUPLEX
    font_clr = (255, 255, 255)
    font_pt = (4, 12)
    font_sc = 0.4

    # Create output directory if desired.
    if opt.write:
        print('==> Will write outputs to %s' % opt.write_dir)
        if not os.path.exists(opt.write_dir):
            os.makedirs(opt.write_dir)

    print('==> Running Demo.')
    while True:

        start = time.time()

        # Get a new image.
        img, status = vs.next_frame()
        if status is False:
            break

        # Get points and descriptors.
        start1 = time.time()
        pts, desc, heatmap = fe.run(img)
        end1 = time.time()

        # Add points and descriptors to the tracker.
        tracker.update(pts, desc)

        # Get tracks for points which were match successfully across all frames.
        tracks = tracker.get_tracks(opt.min_length)

        # Primary output - Show point tracks overlayed on top of input image.
        out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        tracks[:, 1] /= float(fe.nn_thresh)  # Normalize track scores to [0,1].
        tracker.draw_tracks(out1, tracks)
        if opt.show_extra:
            cv2.putText(out1, 'Point Tracks', font_pt, font, font_sc, font_clr, lineType=16)

        # Extra output -- Show current point detections.
        out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        for pt in pts.T:
            pt1 = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
        cv2.putText(out2, 'Raw Point Detections', font_pt, font, font_sc, font_clr, lineType=16)

        # Extra output -- Show the point confidence heatmap.
        if heatmap is not None:
            min_conf = 0.001
            heatmap[heatmap < min_conf] = min_conf
            heatmap = -np.log(heatmap)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
            out3 = myjet[np.round(np.clip(heatmap * 10, 0, 9)).astype('int'), :]
            out3 = (out3 * 255).astype('uint8')
        else:
            out3 = np.zeros_like(out2)
        cv2.putText(out3, 'Raw Point Confidences', font_pt, font, font_sc, font_clr, lineType=16)

        # Resize final output.
        if opt.show_extra:
            out = np.hstack((out1, out2, out3))
            out = cv2.resize(out, (3 * opt.display_scale * opt.W, opt.display_scale * opt.H))
        else:
            out = cv2.resize(out1, (opt.display_scale * opt.W, opt.display_scale * opt.H))

        # Display visualization image to screen.
        if not opt.no_display:
            cv2.imshow(win, out)
            key = cv2.waitKey(opt.waitkey) & 0xFF
            if key == ord('q'):
                print('Quitting, \'q\' pressed.')
                break

        # Optionally write images to disk.
        if opt.write:
            out_file = os.path.join(opt.write_dir, 'frame_%05d.png' % vs.i)
            print('Writing image to %s' % out_file)
            cv2.imwrite(out_file, out)

        end = time.time()
        net_t = (1. / float(end1 - start))
        total_t = (1. / float(end - start))
        if opt.show_extra:
            print('Processed image %d (net+post_process: %.2f FPS, total: %.2f FPS).' \
                  % (vs.i, net_t, total_t))

    # Close any remaining windows.
    cv2.destroyAllWindows()

    print('==> Finshed Demo.')
