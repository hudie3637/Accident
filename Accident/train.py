# For relative import
import os
import sys

from GTAMN import GTAMN
from accident import Accident

from fusion_graph import gatedFusion

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)
print(PROJ_DIR)
import argparse


from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
api_key = "6904b03128d41d459604351aa46ecd82866be0a8"
wandb.login(key=api_key)

from util import *


parser = argparse.ArgumentParser()
args = parser.parse_args()

gpu_num = 0                                                 # set the GPU number of your server.
os.environ['WANDB_MODE'] = 'online'                        # select one from ['online','offline']

def hyperparameter_defaults():
    return {
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
            'intra_M':10 ,  # 您需要确定每个图内部的特征数量
            'intra_d': 10,  # 每个图内部的特征维度
            'inter_M': 24,  # 跨图的特征数量
            'inter_d': 6,  # 跨图的特征维度
            'bn_decay': 0.1  # 批量归一化的衰减率
        }
    }

wandb_proj = 'GTAMN'
wandb.init(config=hyperparameter_defaults, project=wandb_proj)
wandb_logger = WandbLogger()
config = wandb.config


pl.seed_everything(config['train']['seed'])


gpu_id = config['server']['gpu_id']
device =  torch.device("cpu")
# 实例化 gatedFusion
fusion_model = gatedFusion(
    intra_M=wandb.config.fusion['intra_M'],
    intra_d=wandb.config.fusion['intra_d'],
    inter_M=wandb.config.fusion['inter_M'],
    inter_d=wandb.config.fusion['inter_d'],
    bn_decay=wandb.config.fusion['bn_decay']
)
# 加载图数据
road_graph = np.load('Chicago_road.npy')
closeness_graph = np.load('Chicago_closeness.npy')
propagation_graph = np.load('Chicago_propagation_graphs.npy')
intra_M = config['fusion']['intra_M']
# 将图数据转换为 PyTorch 张量并移动到适当的设备上
X1 = torch.tensor(road_graph).to(device)  # 假设 road_graph 是特征数据
X2 = torch.tensor(closeness_graph).to(device)
X3 = torch.tensor(propagation_graph).to(device)

# 根据需要准备 GE1, GE2, GE3，这里使用随机数据作为示例
GE1 = torch.rand(size=(X1.shape[0], intra_M)).to(device)
GE2 = torch.rand(size=(X2.shape[0], intra_M)).to(device)
GE3 = torch.rand(size=(X3.shape[0], intra_M)).to(device)

# 调用 forward 方法进行图融合
fused_output = fusion_model(X1, X2, X3, GE1, GE2, GE3)
# 将融合后的图转换为 PyTorch Tensor
combined_graph_tensor = torch.tensor(fused_output)

class LightningData(LightningDataModule):
    def __init__(self, train_set, val_set, test_set):
        super().__init__()
        self.batch_size = config['train']['batch_size']
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

class LightningModel(LightningModule):
    def __init__(self, scaler, fusiongraph):
        super().__init__()

        # 标准化器，用于数据的标准化和反标准化
        self.scaler = scaler
        # 融合图模型，用于处理图结构数据
        self.fusiongraph = fusiongraph

        # 初始化用于测试的指标计算类
        self.metric_lightning = LightningMetric()

        # 定义损失函数，这里使用的是平均绝对误差（L1 Loss）
        self.loss = nn.L1Loss(reduction='mean')

        self.model = GTAMN(
                gpu_id, fusiongraph,

            )

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
        # 配置模型的优化器
        return Adam(self.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])

def main():
    fusiongraph = gatedFusion(intra_M=..., intra_d=..., inter_M=..., inter_d=..., bn_decay=...)
    # 创建数据集实例
    train_set = Accident(data_dir='path_to_train_data', data_type='train')
    val_set = Accident(data_dir='path_to_val_data', data_type='val')
    test_set = Accident(data_dir='path_to_test_data', data_type='test')
    # 创建 LightningData 和 LightningModel 实例
    lightning_data = LightningData(train_set, val_set, test_set)
    lightning_model = LightningModel(scaler=train_set.scaler, fusiongraph=fusiongraph)

    trainer = Trainer(
        logger=wandb_logger,
        gpus=[gpu_id],
        max_epochs=config['train']['epoch'],
        # TODO
        # precision=16,
    )

    trainer.fit(lightning_model, lightning_data)
    trainer.test(lightning_model, datamodule=lightning_data)



if __name__ == '__main__':
    main()
