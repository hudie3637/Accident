import os
import sys
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.optim as optim
from torch import nn
from torch_geometric.data import DataLoader
from transformers.models.longformer.convert_longformer_original_pytorch_lightning_to_pytorch import LightningModel
from pytorch_lightning import loggers as pl_loggers
from GTAMN import GTAMN
from dataloader import GraphDataset

from fusion_graph import gatedFusion, GEmbedding
import numpy as np
import torch
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
import pytorch_lightning as pl

from util import LightningMetric, masked_mae

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)

# 定义超参数
config = {
    'train': {
        'seed': 42,
        'batch_size': 32,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'epoch': 10
    },
    'server': {
        'gpu_id': 0
    },
    'fusion': {
        'intra_M': 8860,
        'intra_d': 8860,
        'inter_M': 8860,
        'inter_d': 8860,
        'bn_decay': 0.1
    },
    'model': {
        'input_size': 128,  # 示例值，您需要根据模型实际需要设置
        'hidden_size': 256,  # 示例值
        'output_size': 64,  # 示例值
        # ... 其他模型相关配置 ...
    }
}

# 设置随机种子
pl.seed_everything(config['train']['seed'])

# 确定设备
gpu_id = config['server']['gpu_id']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化 gatedFusion
fusion_model = gatedFusion(
    intra_M=config['fusion']['intra_M'],
    intra_d=config['fusion']['intra_d'],
    inter_M=config['fusion']['inter_M'],
    inter_d=config['fusion']['inter_d'],
    bn_decay=config['fusion']['bn_decay']
)

# 加载图数据
road_graph = np.load('data/Chicago/Chicago_road.npy')
closeness_graph = np.load('data/Chicago/Chicago_closeness.npy')
propagation_graph = np.load('data/Chicago/Chicago_propagation_graphs.npy')

# 将图数据转换为 PyTorch 张量并移动到适当的设备上
X1 = torch.tensor(road_graph, dtype=torch.float32).to(device)
X2 = torch.tensor(closeness_graph, dtype=torch.float32).to(device)
X3 = torch.tensor(propagation_graph, dtype=torch.float32).to(device)

# 实例化 GEmbedding，注意 GE 应该是 [num_vertices, num_graphs] 的形状
GE1 = GEmbedding(D=config['fusion']['intra_d'], bn_decay=config['fusion']['bn_decay']).to(device)
GE2 = GEmbedding(D=config['fusion']['intra_d'], bn_decay=config['fusion']['bn_decay']).to(device)
GE3 = GEmbedding(D=config['fusion']['intra_d'], bn_decay=config['fusion']['bn_decay']).to(device)
# 由于 GE 是 [num_vertices, num_graphs] 的形状，我们需要将其转换为相应的形状
GE1_input = X1[..., 0].clone().detach().view(-1, X1.shape[1]).to(torch.long)
GE2_input = X2[..., 0].clone().detach().view(-1, X2.shape[1]).to(torch.long)
GE3_input = X3[..., 0].clone().detach().view(-1, X3.shape[1]).to(torch.long)
# 调用 forward 方法进行图融合
GE1_output = GE1(GE1_input)
GE2_output = GE2(GE2_input)
GE3_output = GE3(GE3_input)

fused_output = fusion_model(X1, X2, X3, GE1_output, GE2_output, GE3_output)
combined_graph_tensor = fused_output.detach().cpu()  # 确保结果可以被打印

class LightningData(pl.LightningDataModule):
    def __init__(self, train_set, val_set, test_set, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=0,
                          pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0,
                          pin_memory=True, drop_last=True)


class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # 从config中提取模型参数
        input_size = config['model']['input_size']
        hidden_size = config['model']['hidden_size']
        output_size = config['model']['output_size']
        intra_M = config['model']['intra_M']
        intra_d = config['model']['intra_d']
        inter_M = config['model']['inter_M']
        inter_d = config['model']['inter_d']
        bn_decay = config['model']['bn_decay']

        # 初始化用于测试的指标计算类
        self.metric_lightning = LightningMetric()

        # 定义损失函数，这里使用的是平均绝对误差（L1 Loss）
        self.loss = nn.L1Loss(reduction='mean')

        # 初始化 GTAMN 模型
        self.model = GTAMN(input_size, hidden_size, output_size, intra_M, intra_d, inter_M, inter_d, bn_decay)
        self.criterion = nn.MSELoss()

        # 对模型参数进行初始化
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # 对非标量参数使用 Xavier 均匀分布初始化
            else:
                nn.init.uniform_(p)  # 对标量参数使用均匀分布初始化

        # 记录配置到日志
        self.log_dict(config)

    def forward(self, x):
        # 定义模型的前向传播过程
        return self.model(x)

    def _run_model(self, batch):
        # 辅助函数，用于运行模型并计算损失
        x, y = batch
        y_hat = self(x)

        # 对预测结果进行反标准化
        y_hat = self.scaler.inverse_transform(y_hat)

        # 计算损失，这里使用的是 masked_mae 函数
        loss = masked_mae(y_hat, y, 0.0)

        return y_hat, y, loss

    def training_step(self, batch, batch_idx):
        # 定义单个训练步骤
        y_hat, y, loss = self._run_model(batch)
        # 记录训练损失
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 定义单个验证步骤
        y_hat, y, loss = self._run_model(batch)
        # 记录验证损失
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        # 定义单个测试步骤
        y_hat, y, loss = self._run_model(batch)
        # 使用 LightningMetric 类计算指标
        self.metric_lightning(y_hat.cpu(), y.cpu())
        # 记录测试损失
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        # 在每个测试周期结束时计算并记录测试指标
        test_metric_dict = self.metric_lightning.compute()
        self.log_dict(test_metric_dict)

    def configure_optimizers(self):
        # 使用 PyTorch 的 Adam 优化器
        optimizer = optim.Adam(self.parameters(), lr=config['train']['lr'],
                               weight_decay=config['train']['weight_decay'])
        return optimizer

def main():
    # 加载数据集
    train_dataset = GraphDataset(graph_dir=os.path.join(PROJ_DIR, 'data', 'Chicago', 'train'))
    val_dataset = GraphDataset(graph_dir=os.path.join(PROJ_DIR, 'data', 'Chicago', 'val'))
    test_dataset = GraphDataset(graph_dir=os.path.join(PROJ_DIR, 'data', 'Chicago', 'test'))

    # 创建 LightningData 实例
    data_module = LightningData(train_dataset, val_dataset, test_dataset, batch_size=config['train']['batch_size'])

    # 创建 LightningModel 实例
    model = LightningModel(**config['model'])

    # 初始化 PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=config['train']['epoch'],
        gpus=[config['server']['gpu_id']],  # 指定 GPU ID
        progress_bar_refresh_rate=20,
        logger=pl_loggers.TensorBoardLogger('logs/', name='GTAMN_logs'),  # 日志保存位置
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=True)],  # 提前停止训练
        deterministic=True,  # 使结果可重复
        precision=16 if torch.cuda.is_available() else 32  # 使用半精度以节省内存
    )

    # 训练模型
    trainer.fit(model, data_module)

    # 验证模型
    trainer.validate(model, datamodule=data_module)

    # 测试模型
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()