import numpy as np

# 定义MLP参数
def mlp_forward(embedding, W1, b1, W2, b2):
    # 第一层
    hidden = np.maximum(0, np.dot(embedding, W1) + b1)  # 使用ReLU激活函数
    # 输出层
    output = np.dot(hidden, W2) + b2
    # 重塑为矩阵
    propagation_matrix = output.reshape((int(np.sqrt(len(output))), int(np.sqrt(len(output)))))
    return propagation_matrix

# 加载事故嵌入特征的NumPy数组
# accident_embeddings = np.load('../data/Chicago/Chicago_accident.npy')
accident_embeddings = np.load('../data/Chicago/Chicago_accident.npy')
# 嵌入维度
Dp = accident_embeddings.shape[1]  # 输入维度

# 定义MLP参数
Wproj1 = np.random.normal(0, 0.1, (Dp, Dp))  # Wproj1的维度应为(Dp, Dp)
bproj1 = np.random.normal(0, 0.1, Dp)  # bproj1的维度应为(Dp,)
Wproj2 = np.random.normal(0, 0.1, (Dp, Dp))  # Wproj2的维度应为(Dp, Dp)
bproj2 = 0.1  # 偏差通常初始化为小常数

# 计算所有嵌入的平均值以获得一个代表性的嵌入
representative_embedding = np.mean(accident_embeddings, axis=0)

# 使用代表性的嵌入生成传播图
propagation_matrix = mlp_forward(representative_embedding, Wproj1, bproj1, Wproj2, bproj2)

# 确保传播图是方阵
assert propagation_matrix.shape[0] == propagation_matrix.shape[1], "传播图不是方阵"

# 保存传播图矩阵
np.save('../data/Chicago/Chicago_propagation_graph.npy', propagation_matrix)