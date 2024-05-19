import torch.nn as nn
import torch

class AccidentSeverityModel(nn.Module):
    def __init__(self, input_feature_size, output_class_size, conv1d_kernel_size, conv1d_num_filters, traffic_data,
                 num_days, num_hours, num_regions, embedding_dim):
        super(AccidentSeverityModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=conv1d_num_filters, kernel_size=conv1d_kernel_size,
                                stride=1, padding=(conv1d_kernel_size - 1) // 2)
        self.conv1d_output_feature_size = conv1d_num_filters * traffic_data.shape[1]
        self.bn1 = nn.BatchNorm1d(conv1d_num_filters)  # 添加批量归一化层

        self.embed_weekday = nn.Embedding(num_days, embedding_dim)
        self.embed_time = nn.Embedding(num_hours, embedding_dim)
        self.embed_region = nn.Embedding(num_regions, embedding_dim)
        # print(f' self.embed_weekday{ self.embed_weekday}')
        # print(f' self.embed_time{self.embed_time}')
        # print(f' self.embed_region{self.embed_region}')
        # 初始化嵌入层权重
        nn.init.xavier_uniform_(self.embed_weekday.weight)
        nn.init.xavier_uniform_(self.embed_time.weight)
        nn.init.xavier_uniform_(self.embed_region.weight)
        # 计算总的输入特征大小，包括嵌入层的特征
        total_embedding_feature_size = embedding_dim * 3*10
        total_input_feature_size = input_feature_size + self.conv1d_output_feature_size + total_embedding_feature_size
        self.accident_embedding = nn.Linear(total_input_feature_size, 256)  # 调整线性层尺寸
        # print(f' self.accident_embedding{self.accident_embedding}')
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层
        self.predicted_severity = nn.Linear(256, output_class_size)  # 调整线性层尺寸

    def forward(self, x_BERT, x_weekday_indices, x_time_indices, x_location_indices, x_traffic):
        for tensor in [x_BERT, x_weekday_indices, x_time_indices, x_location_indices, x_traffic]:
            if torch.isnan(tensor).any():
                print(f"NaN found in {tensor}")
            # 检查嵌入层权重是否有 NaN
        if torch.isnan(self.embed_weekday.weight).any():
            print("NaN found in embed_weekday weight")

            # 检查 x_weekday_indices 是否有 NaN
        if torch.isnan(x_weekday_indices).any():
            print("NaN found in x_weekday_indices")
        x_weekday_embed = self.embed_weekday(x_weekday_indices)
        x_time_embed = self.embed_time(x_time_indices)
        x_location_embed = self.embed_region(x_location_indices)

        # 将嵌入向量展平
        x_weekday_embed = x_weekday_embed.view(x_weekday_embed.size(0), -1)
        x_time_embed = x_time_embed.view(x_time_embed.size(0), -1)
        x_location_embed = x_location_embed.view(x_location_embed.size(0), -1)

        # 打印嵌入层的输出
        # print(f'x_weekday_embed: {x_weekday_embed}')
        # print(f'x_time_embed: {x_time_embed}')
        # print(f'x_location_embed: {x_location_embed}')
        # 处理交通数据
        x_traffic = x_traffic.unsqueeze(1)
        x_traffic = self.conv1d(x_traffic)
        x_traffic = self.bn1(x_traffic)  # 使用批量归一化

        x_traffic = torch.relu(x_traffic)
        # print(f'Conv1d output: {x_traffic}')
        x_traffic = x_traffic.view(x_traffic.size(0), -1)

        # 展平BERT特征
        x_BERT = x_BERT.view(x_BERT.size(0), -1)
        # print(f'Shapes of data arrays: X_BERT={x_BERT.shape}, X_weekday={x_weekday_embed.shape}, X_time={x_time_embed.shape}, X_location={x_location_embed.shape}, X_traffic={x_traffic.shape}')

        # 将所有特征向量拼接起来
        x = torch.cat((x_traffic, x_BERT, x_weekday_embed, x_time_embed, x_location_embed), dim=1)
        # print(f'Concatenated features: {x}')
        # 通过线性层和Dropout层
        embedding = torch.relu(self.accident_embedding(x))
        embedding = self.dropout(embedding)
        # print(f'Output after dropout: {embedding}')
        # 输出预测
        x = self.predicted_severity(embedding)
        # print(f' 预测x{x}')
        return x, embedding