import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
#(113260, 90)
#(10784, 90)
# 定义感兴趣的字段
fields_of_interest = ['valid', 'tmpf', 'dwpf', 'relh', 'sknt', 'mslp', 'skyc1', 'skyl1']

# 加载数据，并仅选择感兴趣的字段
Chicago_df = pd.read_csv('../data/Chicago/raw/chi_weather.csv', usecols=fields_of_interest, parse_dates=['valid'])
NYC_df = pd.read_csv('../data/NYC/raw/NYC_weather.csv', usecols=fields_of_interest, parse_dates=['valid'])

# 用 NaN 替换非数值型字符
num_cols = ['tmpf', 'dwpf', 'relh', 'sknt', 'mslp']
for col in num_cols:
    Chicago_df[col] = Chicago_df[col].replace(['T', 'M', 'OVC'], np.nan)
    NYC_df[col] = NYC_df[col].replace(['T', 'M', 'OVC'], np.nan)

# 使用 SimpleImputer 的 mean 策略来填充数值型缺失值
imputer = SimpleImputer(strategy='mean')
Chicago_df[num_cols] = imputer.fit_transform(Chicago_df[num_cols])
NYC_df[num_cols] = imputer.transform(NYC_df[num_cols])  # 注意这里使用transform以确保和Chicago使用相同的均值填充

# 标准化数值型特征
scaler = StandardScaler()
Chicago_df[num_cols] = scaler.fit_transform(Chicago_df[num_cols])
NYC_df[num_cols] = scaler.transform(NYC_df[num_cols])  # 使用相同的scaler确保一致性

# 独热编码分类变量，同时确保两个数据集编码后的列完全匹配
encoder = OneHotEncoder(drop='first', sparse_output=False)
# 使用concat而不是append合并DataFrame以适应未来版本的pandas
encoder.fit(pd.concat([Chicago_df[['skyc1', 'skyl1']], NYC_df[['skyc1', 'skyl1']]], ignore_index=True))

Chicago_encoded = encoder.transform(Chicago_df[['skyc1', 'skyl1']])
NYC_encoded = encoder.transform(NYC_df[['skyc1', 'skyl1']])

# 将编码后的分类变量转换回DataFrame，并设置正确的列名
encoded_cols = encoder.get_feature_names_out(['skyc1', 'skyl1'])
Chicago_df_encoded = pd.DataFrame(Chicago_encoded, columns=encoded_cols)
NYC_df_encoded = pd.DataFrame(NYC_encoded, columns=encoded_cols)

# 重建DataFrame，仅包含处理后的数值型和分类型字段
Chicago_final = pd.concat([Chicago_df.drop(['skyc1', 'skyl1'], axis=1), Chicago_df_encoded], axis=1)
NYC_final = pd.concat([NYC_df.drop(['skyc1', 'skyl1'], axis=1), NYC_df_encoded], axis=1)

# 将DataFrame转换为NumPy数组
Chicago_weather_embeddings_np = Chicago_final.values
NYC_weather_embeddings_np = NYC_final.values
np.save('../data/Chicago/Chicago_weather.npy', Chicago_weather_embeddings_np)
np.save('../data/NYC/NYC_weather.npy', NYC_weather_embeddings_np)

# print(Chicago_final.shape)
# print(NYC_final.shape)
print(NYC_final.columns)