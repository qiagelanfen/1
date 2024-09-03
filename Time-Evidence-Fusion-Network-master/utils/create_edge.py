import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data


# 假设你的数据已经加载到dataframe中




def create_edge(a):
    if a=='electricity.csv':
        df = pd.read_csv('./dataset/electricity/electricity.csv')
    elif a=='1':
        df = pd.read_csv('./dataset/ETT-small/ETTh2.csv')

    # 删除时间列，或者将时间列设为索引
    df.set_index('date', inplace=True)

    # 假设你的数据集是一个 NumPy 数组
    data = df.to_numpy()  # 转换为 NumPy 数组

    # 将数据转置，特征现在作为节点
    transposed_data = data.T  # 形状为 (7, 60000)

    # 计算特征之间的相关系数矩阵
    corr_matrix = np.corrcoef(transposed_data)

    # 设置相关性阈值
    threshold = 0.6

    # 创建边列表
    edges = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > threshold:
                edges.append([i, j])

    # 转换为 PyTorch 张量
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index



