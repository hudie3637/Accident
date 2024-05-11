from typing import Any, Callable, Optional
import torch
import numpy as np

from torch_geometric.utils import dense_to_sparse, get_laplacian, to_dense_adj
from torchmetrics import Metric


def get_L(W):
    edge_index, edge_weight = dense_to_sparse(W)
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight)
    adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
    return adj


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    # 确保 pred 和 real 的尺寸一致
    print(f'pred {pred.shape},real {real.shape}')


    mae = masked_mae(pred.flatten(), real.flatten(), 0.0).item()
    mape = masked_mape(pred.flatten(), real.flatten(), 0.0).item()
    rmse = masked_rmse(pred.flatten(), real.flatten(), 0.0).item()
    return np.round(mae, 4), np.round(mape, 4), np.round(rmse, 4)

class LightningMetric(Metric):
    def __init__(self):
        super().__init__()
        # 初始化为 Tensor，使用 torch.zeros 来预留空间（如果需要）
        # 这里的尺寸需要根据你的数据进行调整
        self.add_state("y_true", torch.zeros((0,), dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("y_pred", torch.zeros((0,), dtype=torch.float32), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # 直接使用 torch.cat 来累积数据
        # 假设 preds 和 target 的第一个维度是批次大小 (batch_size)
        if self.y_pred.numel() == 0:  # 如果之前没有任何数据，直接赋值
            self.y_pred = preds.clone()
            self.y_true = target.clone()
        else:
            self.y_pred = torch.cat((self.y_pred, preds), dim=0)
            self.y_true = torch.cat((self.y_true, target), dim=0)

    def compute(self):
        if self.y_pred.numel() == 0 or self.y_true.numel() == 0:
            # 如果没有累积任何数据，则返回空的指标字典
            return {}

        # 将累积的数据转换为需要的形状
        y_pred = self.y_pred.view(-1, self.y_pred.size(-1))
        y_true = self.y_true.view(-1, self.y_true.size(-1))

        # 初始化指标字典和列表以计算平均值
        metric_dict = {}
        rmse_avg = []
        mae_avg = []
        mape_avg = []

        # 假设我们有多个时间步长的数据
        for i in range(y_pred.size(1)):
            mae, mape, rmse = metric(y_pred[:, i], y_true[:, i])
            idx = i + 1
            metric_dict.update({'rmse_%s' % idx: rmse})
            metric_dict.update({'mae_%s' % idx: mae})
            metric_dict.update({'mape_%s' % idx: mape})

            rmse_avg.append(rmse)
            mae_avg.append(mae)
            mape_avg.append(mape)

        # 计算并添加平均指标
        metric_dict.update({'rmse_avg': np.mean(rmse_avg)})
        metric_dict.update({'mae_avg': np.mean(mae_avg)})
        metric_dict.update({'mape_avg': np.mean(mape_avg)})

        # 重置状态以准备下一次计算
        self.reset()

        return metric_dict

    def reset(self):
        # 重置累积的预测和真实值张量
        self.y_pred = torch.empty(0, device=self.device)
        self.y_true = torch.empty(0, device=self.device)
class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):

            return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


if __name__ == '__main__':

    lightning_metric = LightningMetric()
    batches = 10
    for i in range(batches):
        preds = torch.randn(32, 24, 38, 1)
        target = preds + 0.15

        rmse_batch = lightning_metric(preds, target)
        print(f"Metrics on batch {i}: {rmse_batch}")

    rmse_epoch = lightning_metric.compute()
    print(f"Metrics on all data: {rmse_epoch}")

    lightning_metric.reset()
