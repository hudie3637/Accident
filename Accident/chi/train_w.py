# For relative import
import os
import sys
from pytorch_lightning.callbacks import ModelCheckpoint

from matplotlib import pyplot as plt, dates

from GTAMN import GTAMN_submodule
from accident import Accident, AccidentGraph
from fusion_graph import FusionGraphModel

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)
print(PROJ_DIR)
import argparse
from pytorch_lightning import seed_everything

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger


from util import *


parser = argparse.ArgumentParser()
args = parser.parse_args()

gpu_num = 0                                                 # set the GPU number of your server.


hyperparameter_defaults = dict(
    server=dict(
        gpu_id=0,
    ),
    graph=dict(
        use=['road', 'closeness', 'pro'],
        fix_weight=False,                                   # if True, the weight of each graph is fixed.
        matrix_weight=True,                                 # if True, turn the weight matrices trainable.
        attention=True,                                    # if True, the SG-ATT is used.
    ),

    data=dict(
        in_dim=24,
        hist_len=24,
        pred_len=1,
        type='chi',
        hidden_size = 128  ,
    ),

    train=dict(
        seed=10,
        epoch=50,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-4,
        M=24,
        d=6,
        bn_decay=0.1,
    )
)


config = hyperparameter_defaults
#
torch.manual_seed(config['train']['seed'])
gpu_id = config['server']['gpu_id']
device = 'cuda:%d' % gpu_id if torch.cuda.is_available() else 'cpu'

root_dir = 'data'
chi_data_dir = os.path.join(root_dir, 'temporal_data/Chicago')
chi_graph_dir = os.path.join(root_dir, 'Chicago')
train_set = Accident(chi_data_dir, 'train')
val_set = Accident(chi_data_dir, 'val')
test_set = Accident(chi_data_dir, 'test')

graph = AccidentGraph(chi_graph_dir, config['graph'], gpu_id)

scaler = train_set.scaler

class LightningData(LightningDataModule):
    def __init__(self, train_set, val_set, test_set):
        super().__init__()
        self.batch_size = config['train']['batch_size']
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=8,
                                   pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=8,
                                 pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8,
                                  pin_memory=True, drop_last=True)


class LightningModel(LightningModule):
    def __init__(self, scaler, fusiongraph):
        super().__init__()

        self.scaler = scaler
        self.fusiongraph = fusiongraph.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.metric_lightning = LightningMetric()

        self.loss = nn.L1Loss(reduction='mean')
        self.model = GTAMN_submodule(
            gpu_id=gpu_id,
            fusiongraph=fusiongraph,
            in_channels=config['data']['in_dim'],
            len_input=config['data']['hist_len'],
            num_for_predict=config['data']['pred_len'],
            hidden_size=config['data']['hidden_size'],
        )
        self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        # self.log_dict(config)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)

        print(f'LightningModel x{x.shape}')
        return self.model(x)

    def _run_model(self, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x, y = batch
        y = y[:, 0, :, :]
        print(f"y: {y.shape}")
        # 确保输入数据需要梯度
        x = x.requires_grad_().to(device)
        y = y.to(device)
        y_hat = self(x)
        # print(f"Output from model: {y_hat.shape}")

        # 逆变换回原始尺度
        y_hat = self.scaler.inverse_transform(y_hat.detach().cpu())
        # print(f"y_hat after inverse transform: {y_hat.shape}")
        # y = y.squeeze(1)  # 移除第二维

        # print(f"y_hat: {y_hat.shape}")

        # 检查 y 的新形状是否与 y_hat 匹配
        if y.shape != y_hat.shape:
            raise ValueError(f"The shapes of y_hat {y_hat.shape} and y {y.shape} do not match")
        y_hat = torch.tensor(y_hat, device=self.device, requires_grad=True)
        y = y.to(self.device)

        loss = masked_mae(y_hat, y, 0.0)
        return y_hat, y, loss

    def training_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        # print(f"y_hat: {y_hat},y{y}")

        self.metric_lightning(y_hat.cpu(), y.cpu())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        test_metric_dict = self.metric_lightning.compute()
        self.log_dict(test_metric_dict)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])





def main():

    fusiongraph = FusionGraphModel(graph, gpu_id, config['graph'], config['data'], config['train']['M'], config['train']['d'], config['train']['bn_decay'])
    fusiongraph = fusiongraph.to(device)

    lightning_data = LightningData(train_set, val_set, test_set)

    lightning_model = LightningModel(scaler, fusiongraph)
    lightning_model.to(device)

    trainer = Trainer(
        accelerator='gpu',  # 指定使用 GPU
        devices=1,  # 指定使用 1 个设备
        max_epochs=config['train']['epoch'],
        # TODO
        # precision=16,
    )

    trainer.fit(lightning_model, lightning_data)
    trainer.test(lightning_model, datamodule=lightning_data)

    # 打印使用的图和数据配置
    print('Graph USE', config['graph']['use'])
    print('Data', config['data'])


if __name__ == '__main__':
    main()
