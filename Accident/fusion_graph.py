import torch
from scipy.stats import cosine
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
class FC(nn.Module):
    def __init__(self, input_dims, units, activations=None, bn_decay=0.1):
        super(FC, self).__init__()
        self.linear = Linear(input_dims, units)
        self.bn = nn.BatchNorm2d(units)
        self.activations = activations

    def forward(self, x):
        print(f'FC_X 输入形状: {x.shape}')
        x = self.linear(x)
        print(f'linear后形状: {x.shape}')
        x = x.view(x.size(0) * x.size(1), -1,1,1)
        print(f'linear后形状: {x.shape}')
        x = self.bn(x)
        if self.activations is not None:
            x = self.activations(x)
        return x


class EdgeFeatureCalculator:
    def __init__(self, node_features, adjacency_matrix):
        """
        初始化 EdgeFeatureCalculator 类。

        参数:
        - node_features: 形状为 (num_nodes, num_features) 的 NumPy 数组，表示节点特征。
        - adjacency_matrix: 形状为 (num_nodes, num_nodes) 的 NumPy 数组，表示邻接矩阵。
        """
        self.node_features = node_features
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = node_features.shape[0]
        self.num_features = node_features.shape[1]

    def calculate_edge_features(self):
        """
        计算边特征，包括点积、欧氏距离和余弦相似度。
        """
        edge_features = []
        # 遍历邻接矩阵中的边
        for i, j in np.argwhere(self.adjacency_matrix):
            # 点积
            dot_product = np.dot(self.node_features[i], self.node_features[j])
            # 欧氏距离
            euclidean_distance = np.linalg.norm(self.node_features[i] - self.node_features[j])
            # 余弦相似度
            cosine_similarity = cosine(self.node_features[i], self.node_features[j])

            # 将计算结果添加到边特征列表中
            edge_features.append((dot_product, euclidean_distance, cosine_similarity))

        # 将边特征转换为 NumPy 数组
        edge_feature_array = np.array(edge_features)

        return edge_feature_array

class Intra_Attention(nn.Module):  # 视图内注意力机制
    '''
    多图注意力机制。
    X:      [num_vertices, num_graphs, num_vertices, D]
    SGE:    [num_vertices, num_graphs, num_vertices, D]
    M:      number of attention heads
    d:      dimension of each attention outputs
    return: [num_vertices, num_graphs, num_vertices, D]
    返回:
    '''

    def __init__(self, M, d, bn_decay, mask=True):
        super(Intra_Attention, self).__init__()
        self.d = d  # 每个头的输出维度
        self.M = M  # 注意力头数
        D = self.M * self.d  # 输入的总维度

        # 定义全连接层，用于生成查询、键、值和输出特征
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X, GE):
        print(f'Intra_Attention_X 输入形状: {X.shape}')
        print(f"FC_q weight shape: {self.FC_q.linear.weight.shape}")

        # 使用插值调整 X 的最后一个维度以匹配 GE
        X = torch.nn.functional.interpolate(X, size=GE.size(2), mode='nearest')

        print(f'后_X 输入形状: {X.shape}')
        GE = GE.expand(-1, X.size(1), -1)  # 现在 GE 可以安全地扩展

        # 使用插值调整 GE 的最后一个维度以匹配 X
        GE = torch.nn.functional.interpolate(GE, size=X.size(2), mode='nearest')

        print(f'Intra_Attention_GE 输入形状: {GE.shape}')
        # 拼接 X 和 GE
        X = torch.cat((X, GE), dim=-1)
        # X 和 GE 拼接，形成注意力机制的输入
        num_vertex_ = X.shape[0]  # 获取节点的数量
        print(f'cat 后形状: {X.shape}')
        # 通过全连接层计算查询、键和值
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)

        # 将查询、键和值分割成 M 个注意力头，并在批次维度上拼接
        query = torch.cat(torch.split(query, self.M, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.M, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.M, dim=-1), dim=0)

        # 调整维度排列，以适应跨视图间的注意力计算
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        # 计算注意力分数并进行缩放
        attention = torch.matmul(query, key)  # 查询和键的矩阵乘法
        attention /= (self.d ** 0.5)  # 缩放以避免梯度消失

        # softmax 归一化注意力分数
        attention = F.softmax(attention, dim=-1)

        # 使用注意力分数加权值，然后整合所有头的输出
        X = torch.matmul(attention, value)  # 注意力加权的值
        X = X.permute(0, 2, 1, 3)  # 调整维度排列
        X = torch.cat(torch.split(X, num_vertex_, dim=0), dim=-1)  # 整合所有头的输出
        print(f'attention 后形状: {X.shape}')
        X = X.permute(0, 3, 1, 2)
        X = X[:, :, :, 0]

        # 通过最后的全连接层生成最终的节点表示
        X = self.FC(X)

        # 清理中间变量以节省内存
        del query, key, value, attention

        return X  # 返回注意力机制处理后的特征矩阵


class Inter_Attention(nn.Module):  # 视图间注意力机制
    '''
    多图注意力机制。
    X:      [num_vertices, num_graphs, num_vertices, D]
    SGE:    [num_vertices, num_graphs, num_vertices, D]
    M:      number of attention heads
    d:      dimension of each attention outputs
    return: [num_vertices, num_graphs, num_vertices, D]
    '''

    def __init__(self, M, d, bn_decay, mask=True):
        super(Inter_Attention, self).__init__()
        self.d = d  # 每个头的输出维度
        self.M = M  # 注意力头数
        D = self.M * self.d  # 输入的总维度
        self.mask = mask  # 是否使用掩码

        # 定义全连接层，用于生成查询、键、值和输出特征
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims= D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X, GE):
        print(f'Inter_X 输入形状: {X.shape}')
        print(f"FC_q weight shape: {self.FC_q.linear.weight.shape}")

        # 使用插值调整 X 的最后一个维度以匹配 GE
        X = torch.nn.functional.interpolate(X, size=GE.size(2), mode='nearest')

        print(f'后_X 输入形状: {X.shape}')
        GE = GE.expand(-1, X.size(1), -1)  # 现在 GE 可以安全地扩展

        # 使用插值调整 GE 的最后一个维度以匹配 X
        GE = torch.nn.functional.interpolate(GE, size=X.size(2), mode='nearest')

        print(f'Inter_GE 输入形状: {GE.shape}')

        # 拼接 X 和 GE
        X = torch.cat((X, GE), dim=-1)
        # X 和 GE 拼接，形成注意力机制的输入
        num_vertex_ = X.shape[0]  # 获取节点的数量


        # 通过全连接层计算查询、键和值
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)  # 形状变化为 [M * num_vertices, num_graphs, num_vertices, d]

        # 将查询、键和值分割成 M 个注意力头，并在批次维度上拼接
        query = torch.cat(torch.split(query, self.M, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.M, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.M, dim=-1), dim=0)

        # 调整维度排列，以适应跨视图间的注意力计算
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        # 计算注意力分数并进行缩放
        attention = torch.matmul(query, key)  # 查询和键的矩阵乘法
        attention /= (self.d ** 0.5)  # 缩放以避免梯度消失

        # 如果使用掩码，应用掩码以避免不相关的注意力计算
        if self.mask:
            num_vertex = X.shape[0]
            num_step = attention.size(2)
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)  # 下三角掩码
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # 增加两个维度
            print(f'Mask shape before expansion: {mask.shape}')
            mask = mask.expand(attention.size(0), -1, num_step, num_step).to(torch.bool)
            print(f'Mask shape after expansion: {mask.shape}')
            mask = mask.to(torch.bool)
            # 使用掩码将不相关的注意力分数设置为负无穷，这样 softmax 会将它们的概率设置为 0
            attention = torch.where(mask, attention, -2 ** 15 + 1)


        # softmax 归一化注意力分数
        attention = F.softmax(attention, dim=-1)

        # 使用注意力分数加权值，然后整合所有头的输出
        X = torch.matmul(attention, value)  # 注意力加权的值
        X = X.permute(0, 2, 1, 3)  # 调整维度排列
        X = torch.cat(torch.split(X, num_vertex_, dim=0), dim=-1)  # 整合所有头的输出
        print(f'attention 后形状: {X.shape}')
        X = X.permute(0, 3, 1, 2)
        X = X[:, :, :, 0]
        # 通过最后的全连接层生成最终的节点表示
        X = self.FC(X)

        # 清理中间变量以节省内存
        del query, key, value, attention

        return X  # 返回注意力机制处理后的特征矩阵


class gatedFusion(nn.Module):
    def __init__(self, intra_M, intra_d, inter_M, inter_d, bn_decay):
        super(gatedFusion, self).__init__()
        # 初始化视图内注意力机制
        self.intra_attention = Intra_Attention(intra_M, intra_d, bn_decay)
        # 初始化视图间注意力机制
        self.inter_attention = Inter_Attention(inter_M, inter_d, bn_decay)

        # 定义门控单元
        self.gate = nn.Sequential(
            nn.Linear((intra_d + inter_d),  (intra_d + inter_d)),
            nn.Sigmoid(),
            nn.Linear( (intra_d + inter_d),(intra_d + inter_d)),
            nn.Tanh()
        )

        # 定义最终的全连接层，用于融合后的输出
        self.final_fc = nn.Linear(2 * (intra_d + inter_d), intra_d + inter_d)

    def forward(self, X1, X2, X3, GE1, GE2, GE3):

        # 计算每个图的视图内注意力机制的输出
        intra_output1 = self.intra_attention(X1, GE1)
        intra_output2 = self.intra_attention(X2, GE2)
        intra_output3 = self.intra_attention(X3, GE3)

        # 计算需要填充的张量
        padding = torch.zeros((intra_output1.size(0) - intra_output3.size(0), intra_output1.size(1), 1, 1))

        # 拼接填充的张量以匹配大小
        intra_output3 = torch.cat((intra_output3, padding), dim=0)
        # 打印调整后的形状以进行检查
        print(f'intra_output1 shape: {intra_output1.shape}')
        print(f'intra_output2 shape: {intra_output2.shape}')
        print(f'intra_output3 shape: {intra_output3.shape}')

        # 计算每个图的视图间注意力机制的输出
        inter_output1 = self.inter_attention(X1, GE1)
        inter_output2 = self.inter_attention(X2, GE2)
        inter_output3 = self.inter_attention(X3, GE3)

        padding = torch.zeros(( inter_output1 .size(0) - inter_output3.size(0),  inter_output1 .size(1), 1, 1))

        # 拼接填充的张量以匹配大小
        inter_output3 = torch.cat((inter_output3, padding), dim=0)

        # 打印调整后的形状以进行检查
        print(f'inter_output1 shape: {inter_output1.shape}')
        print(f'inter_output2 shape: {inter_output2.shape}')
        print(f'inter_output3 shape: {inter_output3.shape}')

        # 将每个图的三个输出堆叠起来
        intra_stacked = torch.stack((intra_output1, intra_output2, intra_output3), dim=0)
        inter_stacked = torch.stack((inter_output1, inter_output2, inter_output3), dim=0)

        # 将门控单元的输入张量平铺
        intra_flattened = intra_stacked.view(-1, *intra_output1.shape[1:])
        inter_flattened = inter_stacked.view(-1, *inter_output1.shape[1:])
        print(f'intra_flattened shape: { intra_flattened.shape}')
        print(f'inter_flattened shape: {inter_flattened.shape}')
        # 拼接 intra_flattened 和 inter_flattened
        combined_flattened = torch.cat((intra_flattened, inter_flattened), dim=-1)

        # # 检查 self.gate 中的线性层权重形状
        linear_layer_weight_shape = self.gate[0].weight.shape
        print(f'Linear layer weight shape in gate: {linear_layer_weight_shape}')
        # # 计算新的形状
        #
        combined_flattened = combined_flattened.view(-1, combined_flattened .size(1))
        print(f'combined_flattened before combine: {combined_flattened.shape}')
        # 通过门控单元
        gated_output = self.gate(combined_flattened)
        print(f'gated_output before shape: {gated_output.shape}')
        print(f'intra_stacked before shape: {intra_stacked.shape}')
        print(f'inter_stacked before shape: {inter_stacked.shape}')
        gated_output = gated_output.unsqueeze(-1).unsqueeze(-1)




        print(f'gated_output: {gated_output.shape}')
        # print(f'intra_stacked shape: {intra_stacked.shape}')
        # print(f'inter_stacked shape: {inter_stacked.shape}')

        # 使用门控权重来融合两种注意力机制的输出
        fused_output = gated_output * intra_stacked + (1 - gated_output) * inter_stacked

        # 展平融合后的输出并通过最终的全连接层
        fused_output = fused_output.view(-1, self.final_fc.in_features)
        fused_output = self.final_fc(fused_output)

        # 恢复到原始形状
        fused_output = fused_output.view(X1.size(0), X1.size(1), X1.size(2), -1)

        return fused_output
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
        'intra_d': 256,
        'inter_M': 2,
        'inter_d': 256,
        'bn_decay': 0.1
    }
}


# 加载图数据
road_graph = np.load('data/Chicago/Chicago_road.npy')
closeness_graph = np.load('data/Chicago/Chicago_closeness.npy')
propagation_graph = np.load('data/Chicago/Chicago_propagation_graphs.npy')
# 打印图数据的形状以进行调试
print(f"Road graph shape: {road_graph.shape}")
print(f"Closeness graph shape: {closeness_graph.shape}")
print(f"Propagation graph shape: {propagation_graph.shape}")# 8860 代表事故嵌入特征的数量，而 16x16 构成了每个传播图的二维矩阵。

# 将图数据转换为 PyTorch 张量并移动到适当的设备上
X1 = torch.tensor(road_graph, dtype=torch.float32).to(device)
X2 = torch.tensor(closeness_graph, dtype=torch.float32).to(device)
X3 = torch.tensor(propagation_graph, dtype=torch.float32).to(device)
#得到多图的嵌入


print(f"X1: {X1.shape}")
print(f"X2 : {X2.shape }")
print(f"X3: {X3.shape}")

X=

print(f"X: {X.shape}")
fusion_model = gatedFusion(
    intra_M=config['fusion']['intra_M'],
    intra_d=config['fusion']['intra_d'],
    inter_M=config['fusion']['inter_M'],
    inter_d=config['fusion']['inter_d'],
    bn_decay=config['fusion']['bn_decay']
).to(device)

# 选择要使用的数据的子集大小
subset_size = 1  # 根据需要调整这个大小

# 仅使用数据的子集
X1_subset = X1[:subset_size]
X2_subset = X2[:subset_size]
X3_subset = X3[:subset_size]


# 使用子集数据调用融合模型
fused_output_subset = fusion_model(X1_subset, X2_subset, X3_subset, GE1_output_subset, GE2_output_subset, GE3_output_subset)


# 将结果移动到 CPU 并打印形状
combined_graph_tensor = fused_output_subset.detach().cpu()
print(f'Fused output shape: {combined_graph_tensor.shape}')