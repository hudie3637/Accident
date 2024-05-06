import numpy as np

# 加载事故嵌入特征的NumPy数组
accident_embeddings = np.load('../data/Chicago/Chicago_accident.npy')

# 嵌入维度和输出维度
Dp = accident_embeddings.shape[1]  # 输入维度
Nv = int(np.sqrt(Dp))  # 假设输出维度是平方数，以便重塑成方阵

# 确保Dp是Nv的平方
assert Dp == Nv**2, "输入维度Dp不是Nv的平方"

# 定义MLP参数
Wproj1 = np.random.normal(0, 0.1, (Dp, Dp))  # Wproj1的维度应为(Dp, Dp)
bproj1 = np.random.normal(0, 0.1, Dp)  # bproj1的维度应为(Dp,)
Wproj2 = np.random.normal(0, 0.1, (Dp, Dp))  # Wproj2的维度应为(Dp, Dp)
bproj2 = 0.1  # 偏差通常初始化为小常数

# 定义激活函数，这里使用ReLU
def relu(x):
    return np.maximum(0, x)

# MLP前向传播以生成传播图
def mlp_forward(embedding, W1, b1, W2, b2):
    # 第一层
    hidden = relu(np.dot(embedding, W1) + b1)
    # 输出层
    output = np.dot(hidden, W2) + b2
    # 重塑为矩阵
    propagation_matrix = output.reshape((Nv, Nv))
    return propagation_matrix

# 对每个嵌入生成传播图
propagation_graphs = np.array([mlp_forward(embed, Wproj1, bproj1, Wproj2, bproj2) for embed in accident_embeddings])

# 保存传播图矩阵
np.save('../data/Chicago/Chicago_propagation_graphs.npy', propagation_graphs)