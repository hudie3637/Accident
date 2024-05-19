# For relative import
import os
import sys

import pandas as pd

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_DIR)


from torch.utils import data
from util import *


class AccidentGraph():

    def __init__(self, graph_dir, config_graph, gpu_id):

        device = 'cuda:%d' % gpu_id if torch.cuda.is_available() else 'cpu'

        use_graph, fix_weight = config_graph['use'], config_graph['fix_weight']



        self.A_road = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'NYC_road.npy')))).to(device)
        self.A_closeness = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'NYC_closeness.npy')))).to(device)
        self.A_pro = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'NYC_propagation_graph.npy')))).to(device)
        self.A_pro =  self.A_pro.unsqueeze(0).unsqueeze(0)

        # 使用双线性插值上采样 X3
        self.A_pro = torch.nn.functional.interpolate(self.A_pro, size=(263, 263), mode='bilinear', align_corners=False)


        self.A_pro =  self.A_pro.squeeze(0).squeeze(0)
        self.node_num = self.A_road.shape[0]

        self.use_graph = use_graph
        self.fix_weight = fix_weight
        self.graph_num = len(use_graph)

    def get_used_graphs(self):
        graph_list = []
        for name in self.use_graph:
            graph_list.append(self.get_graph(name))
        return graph_list

    def get_fix_weight(self):
        return (
               self.A_road * 0.1 + \
               self.A_closeness * 0.5 + \
               self.A_pro * 0.4) / 3

    def get_graph(self, name):
        if name == 'road':
            return self.A_road
        elif name == 'closeness':
            return self.A_closeness
        elif name == 'pro':
            return self.A_pro

        else:
            raise NotImplementedError

class Accident(data.Dataset):
    def __init__(self, data_dir, data_type):
        assert data_type in ['train', 'val', 'test']
        self.data_type = data_type
        self._load_data(data_dir)

    def _load_data(self, data_dir):
        self.data = {}
        for category in ['train', 'val', 'test']:
            file_path = os.path.join(data_dir, f"{category}.npz")
            try:
                cat_data = np.load(file_path, allow_pickle=True)

                # 处理 NaN、inf 和极端值
                self.data['x_' + category] = self._clean_data(cat_data['x'])
                self.data['y_' + category] = self._clean_data(cat_data['y'])

                # # 打印数据集统计信息
                # print(f"{category} 数据集的 x 均值: {self.data['x_' + category][..., 0].mean()}")
                # print(f"{category} 数据集的 x 标准差: {self.data['x_' + category][..., 0].std()}")
            except FileNotFoundError:
                print(f"文件未找到：{file_path}")
                continue

        if 'x_train' in self.data:
            x_train = self.data['x_train'][..., 0]

            # 计算均值和标准差
            mean = x_train.mean()
            std = x_train.std()

            # 避免标准差为零的情况
            if std == 0:
                print("警告: 标准差为零，可能导致标准化错误")
                std = 1  # 避免除以零

            self.scaler = StandardScaler(mean=mean, std=std)
            for category in ['train', 'val', 'test']:
                self.data['x_' + category][..., 0] = self.scaler.transform(self.data['x_' + category][..., 0])

            self.x, self.y = self.data['x_%s' % self.data_type], self.data['y_%s' % self.data_type]

    def _clean_data(self, data):
        # 将 NaN 转换为 0
        data = np.nan_to_num(data)
        # 将 inf 和 -inf 转换为有限的值
        data[np.isinf(data)] = np.nan
        data = np.nan_to_num(data)
        # 设置阈值裁剪极端值
        threshold = 1e6  # 设置一个合理的阈值
        data = np.clip(data, -threshold, threshold)
        return data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

if __name__ == '__main__':
    graph_dir = 'data/NYC'
    config_graph = {
        'use': ['road', 'closeness', 'pro'],
        'fix_weight': True
    }
    gpu_id = 0
    graph = AccidentGraph(graph_dir, config_graph, gpu_id)

    data_dir = 'data/temporal_data/NYC'
    for data_type in ['train', 'val', 'test']:
        file_path = os.path.join(data_dir, f"{data_type}.npz")
        with np.load(file_path, allow_pickle=True) as data:
            print(f"{data_type} dataset: x shape = {data['x'].shape}, y shape = {data['y'].shape}")
    for data_type in ['train', 'val', 'test']:
        file_path = os.path.join(data_dir, f"{data_type}.npz")
        with np.load(file_path, allow_pickle=True) as data:
            print(f"{data_type} dataset: x shape = {data['x'].shape}, y shape = {data['y'].shape}")
    for data_type in ['train', 'val', 'test']:
        dataset = Accident(data_dir, data_type)  # 初始化数据集
        print(f"Length of {data_type} dataset: {len(dataset)}")