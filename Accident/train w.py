# For relative import
import os
import sys

from GTAMN import GTAMN_submodule
from accident import Accident, AccidentGraph
from fusion_graph import FusionGraphModel

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)
print(PROJ_DIR)
import argparse


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
        in_dim=1,
        out_dim=1,
        hist_len=24,
        pred_len=24,
        type='chi',
        hidden_size = 128  ,
        num_rnn_layers = 2
    ),

    train=dict(
        seed=0,
        epoch=40,
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
# pl.utilities.seed.seed_everything(config['train']['seed'])

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

        self.scaler = scaler
        self.fusiongraph = fusiongraph

        self.metric_lightning = LightningMetric()

        self.loss = nn.L1Loss(reduction='mean')
        self.model = GTAMN_submodule(
            gpu_id=gpu_id,
            fusiongraph=fusiongraph,
            in_channels=config['data']['in_dim'],
            len_input=config['data']['hist_len'],
            num_for_predict=config['data']['pred_len'],
            hidden_size=config['data']['hidden_size'],
            num_rnn_layers=config['data']['num_rnn_layers']
        )
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        # self.log_dict(config)

    def forward(self, x):
        return self.model(x)

    def _run_model(self, batch):
        x, y = batch
        print(f'x{x.shape}')
        y_hat = self(x)

        y_hat = self.scaler.inverse_transform(y_hat)

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
        self.metric_lightning(y_hat.cpu(), y.cpu())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        test_metric_dict = self.metric_lightning.compute()
        self.log_dict(test_metric_dict)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])


def main():
    fusiongraph = FusionGraphModel(graph, gpu_id, config['graph'], config['data'], config['train']['M'], config['train']['d'], config['train']['bn_decay'])

    lightning_data = LightningData(train_set, val_set, test_set)

    lightning_model = LightningModel(scaler, fusiongraph)

    trainer = Trainer(
        accelerator='cpu',
        # gpus=[gpu_id],
        max_epochs=config['train']['epoch'],
        # TODO
        # precision=16,
    )

    trainer.fit(lightning_model, lightning_data)
    trainer.test(lightning_model, datamodule=lightning_data)
    print('Graph USE', config['graph']['use'])
    print('Data', config['data'])


if __name__ == '__main__':
    main()
