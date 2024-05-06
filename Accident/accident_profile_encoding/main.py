import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import os

from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader

from model import AccidentSeverityModel
from dataloader import get_data_loaders, custom_collate_fn
from train import train_model, evaluate_model
from utils import load_data
import argparse
# 定义命令行参数解析
parser = argparse.ArgumentParser(description='Accident Severity Prediction')
parser.add_argument('--data_path', type=str, default='../data', help='Path to the data directory')
parser.add_argument('--enable_cuda', action='store_true', help='Enable CUDA')
parser.add_argument('--load_weights', type=str, default=None, help='Path to the model weights file')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
args = parser.parse_args()
traffic_data = np.load('../data/Chicago/Chicago_traffic_weather_sequence.npy')
conv1d_kernel_size = 3
conv1d_num_filters = 64
num_days = 7
num_hours = 24
num_regions = 80
embedding_dim = 64
def main():
    # 根据命令行参数选择设备
    device = torch.device('cuda' if args.enable_cuda and torch.cuda.is_available() else 'cpu')

    # 加载数据
    X_BERT, X_weekday, X_time, X_location, X_traffic, y = load_data(os.path.join(args.data_path, 'Chicago'))
    print(
        f'Shapes of data arrays: X_BERT={X_BERT.shape}, X_weekday={X_weekday.shape}, X_time={X_time.shape}, X_location={X_location.shape}, X_traffic={X_traffic.shape}')

    if X_BERT is None:
        return  # 如果加载数据失败，则退出程序
        # 转换星期几、时间、地点数据为整数索引
        # 确定每个位置编码的最小负值的绝对值加1
    offset_weekday = -X_weekday.min() + 1
    offset_time = -X_time.min() + 1
    offset_location = -X_location.min() + 1

        # 转换位置编码为非负整数索引
    X_weekday_indices = X_weekday + offset_weekday
    X_time_indices = X_time + offset_time
    X_location_indices = X_location + offset_location

    X_weekday_indices = X_weekday_indices.to(torch.long).to(device)
    X_time_indices = X_time_indices.to(torch.long).to(device)
    X_location_indices =X_location_indices.to(torch.long).to(device)


    # 使用train_test_split进行数据划分
    train_idx, test_idx = train_test_split(range(len(y)), test_size=0.2, random_state=42)

    # 根据索引分割所有数据
    # 这里只迁移分割后的数据到设备上
    X_train_BERT = X_BERT[train_idx].to(device)
    X_train_weekday_indices = X_weekday_indices[train_idx].to(device)
    X_train_time_indices = X_time_indices[train_idx].to(device)
    X_train_location_indices = X_location_indices[train_idx].to(device)
    X_train_traffic = X_traffic[train_idx].to(device)
    y_train = y[train_idx].to(device)

    X_test_BERT = X_BERT[test_idx].to(device)
    X_test_weekday_indices = X_weekday_indices[test_idx].to(device)
    X_test_time_indices = X_time_indices[test_idx].to(device)
    X_test_location_indices = X_location_indices[test_idx].to(device)
    X_test_traffic = X_traffic[test_idx].to(device)
    y_test = y[test_idx].to(device)

    # 创建TensorDataset和DataLoader
    train_dataset = TensorDataset(
        X_train_BERT,
        X_train_weekday_indices,
        X_train_time_indices,
        X_train_location_indices,
        X_train_traffic,
        y_train
    )
    test_dataset = TensorDataset(
        X_test_BERT,
        X_test_weekday_indices,
        X_test_time_indices,
        X_test_location_indices,
        X_test_traffic,
        y_test
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 定义模型参数
    input_feature_size = X_BERT.size(1)
    output_class_size = y_train.unique().size(0) + 1  # 获取类别数量并加1

    # 创建模型实例
    model = AccidentSeverityModel(input_feature_size, output_class_size, conv1d_kernel_size, conv1d_num_filters,
                                  traffic_data, num_days, num_hours, num_regions, embedding_dim).to(device)
    # 加载之前保存的最佳模型权重，如果指定了路径
    if args.load_weights and os.path.isfile(args.load_weights):
        model.load_state_dict(torch.load(args.load_weights, map_location=device))

    # 设置损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(args.num_epochs, model, train_loader, test_loader, criterion, optimizer, device)
    # 评估模型
    evaluate_model(model, test_loader,device,'chi_pre_vs_true.csv')
if __name__ == "__main__":
    main()