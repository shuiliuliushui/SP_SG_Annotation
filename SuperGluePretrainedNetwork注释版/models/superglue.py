# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
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
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    """
        
        channels:各层channel的数量
        do_bn:是否做BatchNorm
    """

    n = len(channels)#对应会构造n-1层MLP
    """
    假设channels = [64 , 128 , 256]
    则可以得到2层mlp，分别输入通道和输出通道如下
      layer         in_channel      out_channel
      第1层          64              128
      第2层          128             256
    """
    layers = []
    for i in range(1, n):
        """class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            in_channels:输入通道
            out_channels:输出通道
            kernel_size:卷积核大小(kernel_size,in_channels)，默认与输入通道一致
            stride:卷积步长
            padding：输入的每一条边补充0的层数
            dilation：卷积核元素之间的间距
            groups：把输入通道分组，然后各自进行卷积，输出
            bias：如果bias=True，添加偏置
        """
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))# 这里Conv1d相当于全连接，为什么这里说1维卷积等价与全连接？
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())#每个卷积层后面都加一个非线性激活函数
    return nn.Sequential(*layers)# 使用list添加模型，需要前面加*


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    # 假设keypoints的位置为[u,v]，x轴横向，y轴纵向，则u属于[0,width],v属于[0,height]
    _, _, height, width = image_shape# 读取Image_shape的后两个维度
    one = kpts.new_tensor(1)# 返回一个1
    size = torch.stack([one*width, one*height])[None]# 得到一个[W,H]的张量，比如[[640., 480.]]
    center = size / 2# 比如[[320., 240.]]
    scaling = size.max(1, keepdim=True).values * 0.7 # 比如[[448.]] = 640*0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :] # (x-W/2)/scaling


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    """
        keypoint_location:[x,y,c],x and y image coordinates as well as a detection confidence c
        Visual descriptors di ∈ RD,D=256
        kpts应该是[batch,number,2]，2代表坐标的维度为2
        """
    """__init__的参数是创建这么一个类的时候需要设置的参数，相当于对类具体化"""

    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        """
       keypoint Encoder 中 MLP的网络结构
       输入的channel[3,32,64,128,256,256]
       输入为[x,y,conf],输出256维度跟描述子保持一致
       +------------+--------+------------+-------------+--------------+
       |  in_size   | kernel | in_channel | out_channel |   out_size   |
       +------------+--------+------------+-------------+--------------+
       | [Number,3] |  1*1   |     3      |      32     |              |
       |            |  1*1   |     32     |      64     |              |
       |            |  1*1   |     64     |     128     |              |
       |            |  1*1   |    128     |     256     |              |
       |            |  1*1   |    256     |     256     | [Number,256] |
       +------------+--------+------------+-------------+--------------+
       最后直接和描述子相加：[Number,256] + [Number,256] = [Number,256]
       +------------+--------+------------+-------------+--------------+
       """
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        # torch.nn.init.constant_(tensor, val):使用val填充张量tensor，这里是把最后一层的bias设为0
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        # 输入本来是[batch,number,2],batch是一组的数据个数，交换之后变成[batch,2,number]
        # score是[batch,number],在1加入一个维度后变成[batch,1,number]，然后与[batch,2,number]拼接在dim=1,得到[batch,3,number]
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]# unsqueeze(N)这个函数主要是对数据维度进行扩充,在位置N加上一个维数为1的维度，比如原来是[3]，当N=0时，得到[1,3]
        return self.encoder(torch.cat(inputs, dim=1))


# 对应计算attention那步，权值Alpha就是prob，q是query，k是key，v是value，参考论文
def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    # 输入为[batch,64,4,number]，因为4-head，所以是64x4=256，把256拆成了4组
    # torch.einsum定义乘积形式
    dim = query.shape[1] # 就是dim / num_heads = 64
    # 对于给定的一个head，每一个n与每一个m对应计算所有dim的乘积和，所以最后输出的维度是hnm
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5 #先开根号后除法
    # 在最后一个维度进行softmax，[b,h,n,m],相当于在同一个head下对于query中某一个对key中所有进行计算的值，做一个softmax
    prob = torch.nn.functional.softmax(scores, dim=-1)
    # 对于value[batch,64,4,number]，给定的head下，每一个value要对应乘以一个权值，这里就是m*m，对应相乘，然后这个就是第i个query对应，总共有n个query，
    # 然后每一个维度都计算一下，所以最后保留了dim
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0 # 当表达式为false时，触发异常
        self.dim = d_model // num_heads # 向下取整的除法
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1) # 相当于Wa+b,这里Conv1d就等价于全连接
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)]) # 就是循环三次copy自己的merge层

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        这里输入的query，key，value都是相当于feature，设图像1对应的为feature1，图像2对应的为feature2
        如果是self，就是feature1,feature1,feature1，或者是feature2,feature2,feature2
        如果是cross，就是feature1,feature2,feature2，或者是feature2,feature1,feature1
        每个feature是通过KeypointEncoder的，size为[batch,256,number]
        """
        batch_dim = query.size(0)
        # 计算出q，k，v，这里先输入[batch,256,number]，输出后reshape为[batch,64,4,number]
        """
        对应论文公式(5)
        query_out = W1 * query_in +b1
        key_out   = W2 * key_in   +b2
        value_out = W3 * value_in +b3
        """
        # zip就是把两个列表对应元素变成元组
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1) # 先将x放入计算然后重新reshape
                             for l, x in zip(self.proj, (query, key, value))]  # 3层merge每层对应一个query或者key或者value
        # x:[batch,64,4,number],这是是4-heads的，下面要聚合成256的
        x, _ = attention(query, key, value)
        # 最后通过一个mlp进行merge，先将x变成[batch,256,number]再放入Conv1d
        # contiguous把tensor变成连续内存，只有连续内存才能用view
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    # 对应论文公式（3），计算MLP([X || m])
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim) # 计算m的
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim]) # 构造两层MLP，为什么是feature*2，因为输入是[x || message]两个都是256维度的特征
        nn.init.constant_(self.mlp[-1].bias, 0.0) # 将最后一层的bias变成0

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """
        x, source -> attn(x, source , source) -> attn(query_in, key_in, value_in)
        """
        message = self.attn(x, source, source)
        # x:[batch,256,number]
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        # 每一层有4个heads，有[self,cross]*9共18层,['self', 'cross'] * 9
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        +---------+---------+---------+---------+
        input-->  |  self_0 |   -->   | cross_0 |   -->   |
                  |  self_1 |   -->   | cross_1 |   -->   |
                  |  self_2 |   -->   | cross_2 |   -->   |
                  |  self_3 |   -->   | cross_3 |   -->   |
                  |  self_4 |   -->   | cross_4 |   -->   |
                  |  self_5 |   -->   | cross_5 |   -->   |
                  |  self_6 |   -->   | cross_6 |   -->   |
                  |  self_7 |   -->   | cross_7 |   -->   |
                  |  self_8 |   -->   | cross_8 |   -->   |
                  |  self_9 |   -->   | cross_9 |   -->   | output
        +---------+---------+---------+---------+
        """
        for layer, name in zip(self.layers, self.names):

            """
            layer(x,source):表示用x生成query，用source生成key和value
            所以对于self层，都是用本图像的数据生成q,k,v
                图像1中：x=desc0，source = desc0
                图像2中：x=desc1，source = desc1
            对于cross层，q用本图像数据生成，k和v用另外图像数据生成
                图像1中：x=desc0，source = desc1
                图像2中：x=desc1，source = desc0
            """
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1) #直接相加聚合
        return desc0, desc1
#################################   以上是基于注意力机制的图网络的主体    #################################
#################################   下面是最优化匹配层(sinkhorn)的主体    #################################
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape # b:batch,m：图像1中特征点个数，n：图像2中特征点个数
    one = scores.new_tensor(1) # 数字1
    ms, ns = (m*one).to(scores), (n*one).to(scores) # 数字m，数字n

    bins0 = alpha.expand(b, m, 1) # [b,m,1]
    bins1 = alpha.expand(b, 1, n) # [b,1,n]
    alpha = alpha.expand(b, 1, 1) # [b,1,1]
    # cat[scores,bins0,-1]->[b,m,n+1]：多加了1列全1
    # cat[bins1,alpha,-1]->[b,1,n+1]：1行全1
    # cat([],1)->[b,m+1,n+1]：多加了1行和1列全1
    """
    对应论文里面加入dustbin的约束,在最右侧和最下方加入1
    +---------+---------+---------+---------+---------+
    |    *    |    *    |    *    |    *    |    1    |
    |    *    |    *    |    *    |    *    |    1    |
    |    *    |    *    |    *    |    *    |    1    |
    |    1    |    1    |    1    |    1    |    1    |
    +---------+---------+---------+---------+---------+
        """
    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log() #在指数域就是 1/(M+N)
    """
    log_mu是期望行和,对应指数域=[1/(M+N),1/(M+N),...,N/(M+N)]
    log_nu是期望列和,对应指数域=[1/(M+N),1/(M+N),...,M/(M+N)]
    本来根据论文，期望行和应该是[1,1,...,N]，期望列和应该是[1,1,...,M]
    这里多除以了M+N，后面又乘回来了，所以是等价的，感觉是为了数值稳定性
    """
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
    #进行sinkhorn求解
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N，等价于在指数域乘以(M+N)
    return Z #问题：这里的输出是指数域结果还是对数域结果？


def arange_like(x, dim: int):
    # x:[b,m]或者[b,n]
    # cumsum(0):第一行不动，将第一行累加到下面每一行
    # -1是说索引值为对应顺序减去1
    # 比如x的shape为[b,m]，则返回[0,1,2,3,...,m-1]
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1
#################################   下面是SuperGlue主体    #################################

class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    #配置参数
    default_config = {
        'descriptor_dim': 256,                  #描述子维度，统一为256
        'weights': 'indoor',                    #权重类型，indoor或者outdoor，两套权重
        'keypoint_encoder': [32, 64, 128, 256], #Encoder的通道数
        'GNN_layers': ['self', 'cross'] * 9,    #GNN的层
        'sinkhorn_iterations': 100,             #sinkhorn的最大迭代次数
        'match_threshold': 0.2,                 #匹配筛选的阈值
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        #定义KeypointEncoder
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
        #定义GNN
        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'])
        #GNN输出后的一个全连接
        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)
        #sinkhorn里面的alpha
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        #权重
        assert self.config['weights'] in ['indoor', 'outdoor']
        path = Path(__file__).parent
        path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        self.load_state_dict(torch.load(str(path)))
        print('Loaded SuperGlue model (\"{}\" weights)'.format(
            self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        """
        1.拿到输入
        2.特征点坐标标准化
        3.进入Keypoint Encoder
        4.进入Attention GNN
        5.进行全连接
        6.计算score
        7.进入最优化匹配层
        8.执行mutual check
        9.输出匹配结果
        """
        #1.拿到输入
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }
        #2.特征点坐标标准化
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        #3.进入Keypoint Encoder
        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])
        #4.进入Attention GNN
        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)
        #5.进行全连接
        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        #6.计算score
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5
        #7.进入最优化匹配层
        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])
        #8.执行mutual check
        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        #gather(1,index)表示在dim=1的维度，按照index的索引取数据
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)#这里作者把sinkhorn的结果变回指数域作为匹配得分
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        #进一步筛选大于阈值的匹配
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        #9.输出匹配结果
        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }
