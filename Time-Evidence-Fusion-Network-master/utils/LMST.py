import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

# 加载数据
#data = pd.read_csv('./dataset/electricity/electricity.csv')
#data = pd.read_csv('./dataset/ETT-small/ETTh1.csv')
#data = pd.read_csv('./dataset/exchange_rate/exchange_rate.csv')
#data = pd.read_csv('./dataset/illness/national_illness.csv')
data = pd.read_csv('./dataset/traffic/traffic.csv')
#data = pd.read_csv('./dataset/weather/weather.csv')


# 将日期转换为日期格式
data['date'] = pd.to_datetime(data['date'])
# 提取年、月、日和小时信息，并添加为新的列
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour

# 将新添加的列移到数据框的前面
cols = ['year', 'month', 'day', 'hour'] + list(data.columns[:-4])
data = data[cols]
data = data.drop(columns=['date'])

# 特征和目标都设为最后n列
num_features = 862
features = data.columns[-num_features:]  # 最后n列作为特征
target = data.columns[-num_features:]  # 目标列和特征列一样

# 数据标准化
scaler_target = MinMaxScaler()
data[target] = scaler_target.fit_transform(data[target])

# 将数据转换为 PyTorch 张量
data = torch.tensor(data.values, dtype=torch.float32)
features_data = data[:, 4:]
features_data = features_data.unsqueeze(1)

# 设置超参数
input_dim = num_features
hidden_dim = 1024  # 你可以调整隐藏层的维度
num_layers = 2  # LSTM的层数
batch_size = 64  # 批量大小

# 创建数据加载器
dataset = TensorDataset(features_data, features_data)  # 这里我们使用相同的数据作为输入和目标，因为我们只是想提取特征
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义 LSTM 模型
class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.to('cuda')

    def forward(self, x):
        # LSTM 的输出包括所有时间步的隐藏状态
        x = x.to('cuda')
        out, _ = self.lstm(x)
        return out  # [batch_size, seq_length, hidden_dim]

# 训练 LSTM 并提取特征
def extract_features(dataloader, input_dim, hidden_dim, num_layers):
    model = LSTMFeatureExtractor(input_dim, hidden_dim, num_layers)
    model.train()
    all_features = []

    for x, _ in dataloader:
        features = model(x)
        all_features.append(features.detach().cpu().numpy())

    # 将所有批次的特征拼接起来
    all_features = np.concatenate(all_features, axis=0)
    return all_features

# 提取特征
def get_features(b, t, c):
    # 提取特征
    features = extract_features(dataloader, input_dim, hidden_dim, num_layers)


    # 重塑特征
    features = features.reshape(features.shape[0], features.shape[2])


    # 创建 PCA 对象
    pca = PCA(n_components=c)  # 使用特征数作为 PCA 组件数


    # 应用 PCA 变换
    features_pca_np = pca.fit_transform(features)


    # 将 PCA 变换后的特征转换回 PyTorch 张量
    features_pca = torch.from_numpy(features_pca_np).float()

    # 定义线性变换
    linear_transform = nn.Linear(c, c)
    linear_transform2 = nn.Linear(features_pca.shape[0], b * t)

    # 应用线性变换
    features_pca = linear_transform(features_pca)
    features_pca = features_pca.permute(1, 0)
    features_pca = linear_transform2(features_pca)
    features_pca = features_pca.permute(1, 0)

    # 重塑特征
    features_pca = features_pca.reshape(b, t, c, 1)

    return features_pca


