import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_graph import gatedFusion


# 定义GRU层以捕获时间依赖性
class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        return output, hidden


# 回归MLP，用于最终的交通动态预测
class RegressionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionMLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# GTAMN模型
class GTAMN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, intra_M, intra_d, inter_M, inter_d, bn_decay):
        super(GTAMN, self).__init__()
        self.mvgf = gatedFusion(intra_M, intra_d, inter_M, inter_d, bn_decay)
        self.gru_layer = GRULayer(input_size, hidden_size)
        self.regression_mlp = RegressionMLP(input_size, hidden_size, output_size)

    def forward(self, x):
        # 多视图融合
        graph_fusion_output = self.mvgf(x)

        # 初始化隐藏状态
        hidden = torch.zeros(graph_fusion_output.size(0), self.gru_layer.gru.hidden_size).to(graph_fusion_output.device)

        # 准备时间序列数据用于GRU
        gru_input = graph_fusion_output.permute(0, 2, 1)

        # GRU层捕获时间依赖性
        gru_output, hidden = self.gru_layer(gru_input, hidden)

        # 取最后一个时间步的输出用于预测
        last_time_step_output = gru_output.permute(0, 2, 1)[:, -1, :]

        # 回归MLP进行预测
        prediction = self.regression_mlp(last_time_step_output)

        return prediction
