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

# 加载预训练的BERT模型和分词器（确保只加载一次）
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 定义函数来批量编码文本，以减少内存使用
def batch_encode_texts(texts, chunk_size=100):
    model.eval()  # 将模型设置为评估模式
    chunked_encoded_texts = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        encoded_chunk = outputs.last_hidden_state[:, 0, :]
        chunked_encoded_texts.append(encoded_chunk.cpu())  # 移动到CPU以减少内存消耗
    return torch.cat(chunked_encoded_texts, dim=0).numpy()

# 读取CSV文件
accident_data = pd.read_csv('../data/NYC/2016_accident/Motor_Vehicle_Collisions_-_Crashes_20240325.csv')
# 数据清洗：填充空字符串代替NaN
accident_data.fillna({'VEHICLE TYPE CODE 1': '', 'VEHICLE TYPE CODE 2': '', 'CONTRIBUTING FACTOR VEHICLE 1': ''}, inplace=True)

# 定义严重程度的划分规则
def map_severity(injured, killed):
    if killed > 0:
        return 3  # 至少有一人死亡
    elif injured >= 3:
        return 2  # 受伤人数大于等于3
    else:
        return 1  # 轻微事故

# 应用严重程度的划分规则
accident_data['Severity'] = accident_data.apply(lambda row: map_severity(row['NUMBER OF PERSONS INJURED'], row['NUMBER OF PERSONS KILLED']), axis=1)

# 删除不必要的列
accident_data.drop(['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED'], axis=1, inplace=True)

# 准备数据
vehicle_type_1 = accident_data['VEHICLE TYPE CODE 1'].tolist()
vehicle_type_2 = accident_data['VEHICLE TYPE CODE 2'].tolist()
accident_factor = accident_data['CONTRIBUTING FACTOR VEHICLE 1'].tolist()

# 批量编码
encoded_vehicle_type_1 = batch_encode_texts(vehicle_type_1)
encoded_vehicle_type_2 = batch_encode_texts(vehicle_type_2)
encoded_accident_factor = batch_encode_texts(accident_factor)

# 将编码后的特征向量拼接起来
BERT_features = np.concatenate((encoded_vehicle_type_1, encoded_vehicle_type_2, encoded_accident_factor), axis=1)

# 保存NumPy数组到.npy文件
np.save('NYC_BERT_features.npy', BERT_features)

y = accident_data['Severity'].values
y_2d = np.expand_dims(y, axis=1)
np.save('../data/NYC/NYC_severity_labels.npy',y_2d)
