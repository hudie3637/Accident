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



        self.A_road = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'Chicago_road.npy')))).to(device)
        self.A_closeness = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'Chicago_closeness.npy')))).to(device)
        self.A_pro = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'Chicago_propagation_graph.npy')))).to(device)

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

                self.data['x_' + category] = cat_data['x']
                self.data['y_' + category] = cat_data['y']
            except FileNotFoundError:
                print(f"文件未找到：{file_path}")
                continue

        # 计算 'x_train' 数据的均值和标准差
        if 'x_train' in self.data:
            # 计算 'x_train' 数据的均值和标准差
            mean = self.data['x_train'][..., 0].mean()
            std = self.data['x_train'][..., 0].std()
            self.scaler = StandardScaler(mean=mean, std=std)
            for category in ['train', 'val', 'test']:
                self.data['x_' + category][..., 0] = self.scaler.transform(self.data['x_' + category][..., 0])
            self.x, self.y = self.data['x_%s' % self.data_type], self.data['y_%s' % self.data_type]
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

if __name__ == '__main__':
    graph_dir = 'data/Chicago'
    config_graph = {
        'use': ['road', 'closeness', 'pro'],
        'fix_weight': True
    }
    gpu_id = 0
    graph = AccidentGraph(graph_dir, config_graph, gpu_id)

    data_dir = 'data/temporal_data/Chicago'
    for data_type in ['train', 'val', 'test']:
        dataset = Accident(data_dir, data_type)  # 初始化数据集
        print(f"Length of {data_type} dataset: {len(dataset)}")
