import os

import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

from accident import Accident, AccidentGraph


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        x = x.to(self.device)
        return self.mlp(x)


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None, device=None):  # 添加 device 参数
        super(conv2d_, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.conv = self.conv.to(self.device)

        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        self.batch_norm = self.batch_norm.to(self.device)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.to(self.device)
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True, device=None):
        super(FC, self).__init__()
        # 确保在构造函数中添加了 device 参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 如果不是列表或元组，将单个值转换为列表
        if isinstance(units, int):
            units = [units]
        if isinstance(input_dims, int):
            input_dims = [input_dims]
        if not isinstance(activations, (list, tuple)):  # 假设activations应该是一个列表或元组
            activations = [activations]
        self.convs = nn.ModuleList([
            conv2d_(
                input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
                padding='VALID', use_bias=use_bias, activation=activation,
                bn_decay=bn_decay, device=self.device  # 传递 device 参数
            ) for input_dim, num_unit, activation in zip(input_dims, units, activations)
        ])

    def forward(self, x):
        x = x.to(self.device)
        for conv in self.convs:
            x = conv(x)
        return x



class GEmbedding(nn.Module):
    """
    multi-graph spatial embedding
    GE:     [num_vertices, num_graphs, 1]
    D:      output dims = M * d
    retrun: [num_vertices, num_graphs, num_vertices, D]
    """

    def __init__(self, D, bn_decay, device):
        super(GEmbedding, self).__init__()
        self.embed_layer = nn.Linear(3, D)
        self.device = device
        self.to(self.device)  # 确保所有参数都在 self.device 上

    def forward(self,  GE):


        graph_embbeding = torch.empty(GE.shape[0], GE.shape[1], 3)
        for i in range(GE.shape[0]):
            graph_embbeding[i] = F.one_hot(GE[..., 0][i].to(torch.int64) % 3, 3)

        GE = graph_embbeding.unsqueeze(2).repeat(1, 1, GE.size(0), 1)
        GE = GE.to(self.device)  # 确保 GE 在正确的设备上
        # print("Device before embedding:", GE.device)
        GE = self.embed_layer(GE)
        # print("Device after embedding:", GE.device)
        # print(f'GE{GE.shape}')
        return GE


class spatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [num_vertices, num_graphs, num_vertices, D]
    GE:    [num_vertices, num_graphs, num_vertices, D]
    M:      number of attention heads
    d:      dimension of each attention outputs
    return: [num_vertices, num_graphs, num_vertices, D]
    '''

    def __init__(self, M, d, bn_decay):
        super(spatialAttention, self).__init__()
        self.d = d
        self.M = M
        D = self.M * self.d
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, GE):


        num_vertex = X.shape[0]
        X = torch.cat((X, GE), dim=-1)

        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)

        query = torch.cat(torch.split(query, self.M, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.M, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.M, dim=-1), dim=0)

        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)

        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, num_vertex, dim=0), dim=-1)
        X = self.FC(X)

        del query, key, value, attention  # explicitly delete tensors to free memory
        torch.cuda.empty_cache()  # clear unused memory

        return X


class graphAttention(nn.Module):
    '''
    multi-graph attention mechanism
    X:      [num_vertices, num_graphs, num_vertices, D]
    SGE:    [num_vertices, num_graphs, num_vertices, D]
    M:      number of attention heads
    d:      dimension of each attention outputs
    return: [num_vertices, num_graphs, num_vertices, D]
    '''

    def __init__(self, M, d, bn_decay, mask=True):
        super(graphAttention, self).__init__()
        self.d = d
        self.M = M
        D = self.M * self.d
        self.mask = mask
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, SGE):
        num_vertex = X.shape[0]
        X = torch.cat((X, SGE), dim=-1)

        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)

        query = torch.cat(torch.split(query, self.M, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.M, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.M, dim=-1), dim=0)

        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)

        if self.mask:
            mask = torch.tril(torch.ones(num_vertex, num_vertex, device=X.device))
            attention = attention.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, num_vertex, dim=0), dim=-1)
        X = self.FC(X)


        del query, key, value, attention  # explicitly delete tensors to free memory
        torch.cuda.empty_cache()  # clear unused memory
        return X


class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [num_vertices, num_graphs, num_vertices, D]
    HG:     [num_vertices, num_graphs, num_vertices, D]
    D:      output dims = M * d
    return: [num_vertices, num_graphs, num_vertices, D]
    '''

    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, HS, HG):
        XS = self.FC_xs(HS)
        XG = self.FC_xt(HG)
        z = torch.sigmoid(XS + XG)
        H = (z * HS) + ((1 - z) * HG)
        H = self.FC_h(H)
        return H


class STAttBlock(nn.Module):
    def __init__(self, M, d, bn_decay, mask=False):
        super(STAttBlock, self).__init__()
        self.spatialAttention = spatialAttention(M, d, bn_decay)
        self.graphAttention = graphAttention(M, d, bn_decay, mask=mask)
        self.gatedFusion = gatedFusion(M * d, bn_decay)

    def forward(self, X, GE):


        HS = self.spatialAttention(X, GE)
        HT = self.graphAttention(X, GE)
        H = self.gatedFusion(HS, HT)
        del HS, HT
        return torch.add(X, H)


class FusionGraphModel(nn.Module):
    def __init__(self, graph, gpu_id, conf_graph, conf_data, M, d, bn_decay):
        super(FusionGraphModel, self).__init__()
        self.M = M
        self.d = d
        self.bn_decay = bn_decay

        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # 将模型移动到 self.device
        D = self.M * self.d
        self.SG_ATT = STAttBlock(M, d, bn_decay)
        self.GEmbedding = GEmbedding(D, bn_decay, device=self.device)  # 创建 GEmbedding 实例
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None], bn_decay=self.bn_decay,
                       use_bias=True, device=self.device)  # 确保传递 device 参数
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None], bn_decay=self.bn_decay,
                       use_bias=True, device=self.device)  # 确保传递 device 参数

        self.graph = graph
        self.matrix_w = conf_graph['matrix_weight']
        # matrix_weight: if True, turn the weight matrices trainable.
        self.attention = conf_graph['attention']
        # attention: if True, the SG-ATT is used.
        self.task = conf_data['type']
        device = 'cuda:%d' % gpu_id

        if self.graph.graph_num == 1:
            self.fusion_graph = False
            self.A_single = self.graph.get_graph(graph.use_graph[0])
        else:
            self.fusion_graph = True
            self.softmax = nn.Softmax(dim=1)

            if self.matrix_w:
                adj_w = nn.Parameter(torch.randn(self.graph.graph_num, self.graph.node_num, self.graph.node_num))
                adj_w_bias = nn.Parameter(torch.randn(self.graph.node_num, self.graph.node_num))
                self.adj_w_bias = nn.Parameter(adj_w_bias.to(device), requires_grad=True)
                self.linear = linear(5, 1)

            else:
                adj_w = nn.Parameter(torch.randn(1, self.graph.graph_num).to(self.device))
            self.adj_w = nn.Parameter(adj_w.to(device), requires_grad=True)
            self.used_graphs = self.graph.get_used_graphs()
            assert len(self.used_graphs) == self.graph.graph_num

    def forward(self):
        if self.graph.fix_weight:
            return self.graph.get_fix_weight()

        if self.fusion_graph:
            if not self.matrix_w:
                self.A_w = self.softmax(self.adj_w)[0]
                adj_list = [self.used_graphs[i] * self.A_w[i] for i in range(self.graph.graph_num)]
                self.adj_for_run = torch.sum(torch.stack(adj_list), dim=0)
                # create a graph stack

            else:
                if self.attention:
                    W = torch.stack((self.used_graphs))
                    GE = W[:, :, 0].permute(1, 0).unsqueeze(dim=2)
                    print("Device before to():", GE.device)


                    GE = GE.to(self.device)  # 确保 GE 在正确的设备上
                    # generate graph embbeding

                    # print("Device after to():", GE.device)

                    GE = self.GEmbedding(GE)
                    # print("Device after GEmbedding:", GE.device)

                    W = self.FC_1(torch.unsqueeze(W.permute(1, 0, 2), -1))
                    W = self.SG_ATT(W, GE)
                    # multi-graph spatial attention

                    W = self.FC_2(W).squeeze(dim=-1)
                    W = torch.sum(self.adj_w * W.permute(1, 0, 2), dim=0)

                else:
                    W = torch.sum(self.adj_w * torch.stack(self.used_graphs), dim=0)
                act = nn.ReLU()
                W = act(W)
                self.adj_for_run = W

        else:
            self.adj_for_run = self.A_single

        return self.adj_for_run


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = {
        'train': {
            'seed': 42,
            'batch_size': 32,
            'lr': 0.001,
            'weight_decay': 0.0001,
            'epoch': 1
        },
        'server': {
            'gpu_id': 0
        },
        'fusion': {
            'intra_M': 2,
            'intra_d': 24,
            'inter_M': 2,
            'inter_d': 24,
            'bn_decay': 0.1
        },
        'graph': {
            'use': ['road', 'closeness', 'pro'],
            'fix_weight': False,
            'matrix_weight': True,
            'attention': True
        },
        'data': {
            'in_dim': 1,
            'out_dim': 1,
            'hist_len': 24,
            'pred_len': 24,
            'type': 'NYC',
        }
    }

    gpu_id = config['server']['gpu_id']
    # 加载图数据
    road_graph = np.load('data/NYC/NYC_road.npy')
    closeness_graph = np.load('data/NYC/NYC_closeness.npy')
    propagation_graph = np.load('data/NYC/NYC_propagation_graph.npy')
    # 打印图数据的形状以进行调试
    print(f"Road graph shape: {road_graph.shape}")
    print(f"Closeness graph shape: {closeness_graph.shape}")
    print(f"Propagation graph shape: {propagation_graph.shape}")

    # 将图数据转换为 PyTorch 张量并移动到适当的设备上
    X1 = torch.tensor(road_graph, dtype=torch.float32).to(device)
    X2 = torch.tensor(closeness_graph, dtype=torch.float32).to(device)
    X3 = torch.tensor(propagation_graph, dtype=torch.float32).to(device)
    # 首先，将 X3 转换为四维张量，添加一个批处理维度和一个通道维度
    X3 = X3.unsqueeze(0).unsqueeze(0)  # 现在 X3 的形状是 [1, 1, 16, 16]

    # 使用双线性插值上采样 X3
    X3 = torch.nn.functional.interpolate(X3, size=(X1.size(0), X1.size(1)), mode='bilinear', align_corners=False)

    # 移除不必要的维度，以匹配 X1 和 X2 的形状
    X3 = X3.squeeze(0).squeeze(0)
    #得到多图的嵌入

    print(f"X1: {X1.shape}")
    print(f"X2 : {X2.shape }")
    print(f"X3: {X3.shape}")
    root_dir = 'data'
    chi_data_dir = os.path.join(root_dir, 'temporal_data/NYC')
    chi_graph_dir = os.path.join(root_dir, 'NYC')
    train_set = Accident(chi_data_dir, 'train')
    val_set = Accident(chi_data_dir, 'val')
    test_set = Accident(chi_data_dir, 'test')
    graph = AccidentGraph(chi_graph_dir, config['graph'], gpu_id)
    # 使用 fusion 部分中的 M 和 d 值
    intra_M = config['fusion']['intra_M']
    intra_d = config['fusion']['intra_d']
    bn_decay = config['fusion']['bn_decay']
    fusiongraph = FusionGraphModel(
        graph=graph,  # 确保 graph 是正确配置的图对象
        gpu_id=gpu_id,
        conf_graph=config['graph'],
        conf_data=config['data'],
        M=intra_M,  # 使用 fusion 部分中的 M 值
        d=intra_d,  # 使用 fusion 部分中的 d 值
        bn_decay=bn_decay  # 使用 fusion 部分中的 bn_decay 值
    )

    GE = torch.randn(X1.size(0), 3, 1)
    GE = GE.to(fusiongraph.device)

    # 调用模型的 forward 方法进行计算
    # 注意：这里的 X 是模型需要的输入，根据您的模型定义，您可能需要修改这个部分
    X = torch.stack((X1, X2, X3))  # 假设您要融合这三个图
    print(f'X.shape{X.shape}')
    print(f'GE.shape{GE.shape}')


    # 调用模型的 forward 方法
    output_adjacency_matrix = fusiongraph()

    # 输出融合后的邻接矩阵
    print(output_adjacency_matrix)
