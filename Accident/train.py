import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt, dates

from GTAMN import GTAMN_submodule
from accident import Accident, AccidentGraph
from fusion_graph import FusionGraphModel
import wandb

# Setup for project directory
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)

# Import utility functions
from util import *

api_key = "6904b03128d41d459604351aa46ecd82866be0a8"
wandb.login(key=api_key)

# Configuration dictionary
config = {
    'server': {'gpu_id': 0},
    'graph': {
        'use': ['road', 'closeness','pro'],
        'fix_weight': False,
        'matrix_weight': True,
        'attention': True
    },
    'data': {
        'in_dim': 24,
        'hist_len': 24,
        'pred_len': 1,
        'type': 'nyc',
        'hidden_size': 128
    },
    'train': {
        'seed': 0,
        'epochs': 100,
        'batch_size': 16,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'M': 24,
        'd': 6,
        'bn_decay': 0.1
    }
}

# Initialize WandB
wandb.init(project='accident_analysis', config=config)
config = wandb.config

#
# pl.utilities.seed.seed_everything(config['train']['seed'])

gpu_id = config['server']['gpu_id']
device = 'cuda:%d' % gpu_id if torch.cuda.is_available() else 'cpu'
torch.manual_seed(config['train']['seed'])

root_dir = 'data'
data_dir = os.path.join(root_dir, 'temporal_data/NYC')
graph_dir = os.path.join(root_dir, 'NYC')

train_set = Accident(data_dir, 'train')
val_set = Accident(data_dir, 'val')
test_set = Accident(data_dir, 'test')

graph = AccidentGraph(graph_dir, config['graph'], gpu_id)


# DataLoader class
class AccidentDataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers, pin_memory, drop_last):
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                     num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)  # This returns the number of batches in the DataLoader


# Model class
class AccidentModel(nn.Module):
    def __init__(self, scaler, fusiongraph):
        super().__init__()
        self.scaler = scaler
        self.fusiongraph = fusiongraph
        self.loss_fn = nn.L1Loss(reduction='mean')
        self.model = GTAMN_submodule(
            gpu_id=config['server']['gpu_id'],
            fusiongraph=fusiongraph,
            in_channels=config['data']['in_dim'],
            len_input=config['data']['hist_len'],
            num_for_predict=config['data']['pred_len'],
            hidden_size=config['data']['hidden_size'],
        )
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)


# Helper functions for metrics
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Training function

def train_model(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x, y = batch
        y = y[:, 0, :, :]
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = model.loss(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        wandb.log({"train_loss": loss.item()})
    return total_loss / len(train_loader)

# Validation function
def validate_model(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            y = y[:, 0, :, :]
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = model.loss(y_hat, y)
            total_loss += loss.item()
            wandb.log({"val_loss": loss.item()})
    return total_loss / len(val_loader)
def test_model(model, test_loader):
    model.eval()
    total_loss = 0
    mae_sum = 0
    mape_sum = 0
    rmse_sum = 0
    num_batches = len(test_loader)
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y = y[:, 0, :, :]
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = model.loss(y_hat, y)
            total_loss += loss.item()

            # Calculate metrics using numpy
            y_np = y.cpu().numpy()
            y_hat_np = y_hat.cpu().numpy()
            mae_sum += mean_absolute_error(y_np, y_hat_np)
            mape_sum += mean_absolute_percentage_error(y_np, y_hat_np)
            rmse_sum += root_mean_squared_error(y_np, y_hat_np)

    avg_loss = total_loss / num_batches
    avg_mae = mae_sum / num_batches
    avg_mape = mape_sum / num_batches
    avg_rmse = rmse_sum / num_batches

    print(f"Average Test Loss: {avg_loss:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average MAPE: {avg_mape:.4f}%")
    print(f"Average RMSE: {avg_rmse:.4f}")
    wandb.log({"test_loss": avg_loss, "mae_avg": avg_mae, "mape_avg": avg_mape, "rmse_avg": avg_rmse})

    return avg_loss, avg_mae, avg_mape, avg_rmse
def main():
    # Data loaders setup
    train_loader = AccidentDataLoader(train_set, config['train']['batch_size'], True, 16, True, True)
    val_loader = AccidentDataLoader(val_set, config['train']['batch_size'], False, 16, True, True)
    test_loader = AccidentDataLoader(test_set, config['train']['batch_size'], False, 16, True, True)

    # Initialize model and optimizer
    fusiongraph = FusionGraphModel(graph, config['server']['gpu_id'], config['graph'], config['data'],
                                   config['train']['M'], config['train']['d'], config['train']['bn_decay'])
    model = AccidentModel(train_set.scaler, fusiongraph).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])

    # Training and validation loop
    for epoch in range(config['train']['epochs']):
        train_loss = train_model(model, train_loader, optimizer)
        val_loss = validate_model(model, val_loader)
        print(
            f"Epoch {epoch + 1}/{config['train']['epochs']}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss_epoch": train_loss, "val_loss_epoch": val_loss})

    # Test phase
    test_loss, mae_avg, mape_avg, rmse_avg = test_model(model, test_loader)
    print(
        f"Final Test Metrics - Loss: {test_loss:.4f}, MAE: {mae_avg:.4f}, MAPE: {mape_avg:.4f}%, RMSE: {rmse_avg:.4f}")


if __name__ == '__main__':
    main()
