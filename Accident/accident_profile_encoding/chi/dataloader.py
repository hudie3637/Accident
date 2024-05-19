import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence


def custom_collate_fn(batch):
    # 假设每个 item 是一个包含6个元素的元组 (X_BERT, X_weekday, X_time, X_location, X_traffic, y)
    batch_X_BERT = torch.stack([item[0] for item in batch], 0)
    batch_X_weekday = torch.stack([item[1] for item in batch], 0)
    batch_X_time = torch.stack([item[2] for item in batch], 0)
    batch_X_location = torch.stack([item[3] for item in batch], 0)
    batch_X_traffic = torch.stack([item[4] for item in batch], 0)
    batch_y = torch.stack([item[5] for item in batch], 0)

    # 返回一个二元组 ((inputs...), labels)
    return (batch_X_BERT, batch_X_weekday, batch_X_time, batch_X_location, batch_X_traffic), batch_y
def get_data_loaders(X_BERT, X_weekday, X_time, X_location, X_traffic, y, batch_size):
    # 创建TensorDataset
    dataset = TensorDataset(X_BERT, X_weekday, X_time, X_location, X_traffic, y)

    # 划分数据集
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset, test_dataset = dataset[train_idx], dataset[test_idx]

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, test_loader