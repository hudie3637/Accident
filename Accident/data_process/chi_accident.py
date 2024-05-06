import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

# 禁用符号链接警告
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 定义函数来批量编码文本
def batch_encode_texts(texts, chunk_size=100):
    model.eval()
    encoded_texts = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        encoded_chunk = outputs.last_hidden_state[:, 0, :]
        encoded_texts.append(encoded_chunk.cpu())
    return torch.cat(encoded_texts, dim=0).numpy()

# 读取CSV文件
accident_data = pd.read_csv('../data/Chicago/2016_accident/Traffic_Crashes.csv')

# 填充空值
accident_data.fillna({'INJURIES_NON_INCAPACITATING': 0, 'INJURIES_INCAPACITATING': 0, 'PRIM_CONTRIBUTORY_CAUSE': '未知'}, inplace=True)

# 定义严重程度的划分规则
def map_severity(injured, killed):
    injured = int(injured) if not pd.isna(injured) else 0
    killed = int(killed) if not pd.isna(killed) else 0

    if killed > 0:
        return 3
    elif injured >= 3:
        return 2
    else:
        return 1

# 应用严重程度的划分规则
accident_data['Severity'] = accident_data.apply(lambda row: map_severity(row['INJURIES_NON_INCAPACITATING'], row['INJURIES_INCAPACITATING']), axis=1)

# 删除不必要的列
accident_data.drop(['INJURIES_INCAPACITATING', 'INJURIES_NON_INCAPACITATING'], axis=1, inplace=True)

# 准备数据
text_columns = ['PRIM_CONTRIBUTORY_CAUSE']
text_data = accident_data['PRIM_CONTRIBUTORY_CAUSE'].tolist()

# 使用BERT模型对文本进行编码
encoded_texts = batch_encode_texts(text_data)
np.save('../data/Chicago/Chicago_BERT_features.npy', encoded_texts)

y = accident_data['Severity'].values
np.save('../data/Chicago/Chicago_severity_labels.npy', y)
