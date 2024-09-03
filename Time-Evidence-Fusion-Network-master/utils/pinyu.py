import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
#data = pd.read_csv('./dataset/electricity/electricity.csv')
#data = pd.read_csv('./dataset/ETT-small/ETTh1.csv')
#data = pd.read_csv('./dataset/ETT-small/ETTh2.csv')
#data = pd.read_csv('./dataset/ETT-small/ETTm1.csv')
#data = pd.read_csv('./dataset/ETT-small/ETTm2.csv')
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
data=data.drop(columns=['date'])
# 提取特征和目标值
features = data.iloc[:, 3:].values  # 假设前四列是日期时间
target = data.iloc[:, 3:].values
# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

def compute_fft_features(data):
    num_samples, num_features = data.shape

    magnitude_features = np.zeros((num_samples, num_features))
    phase_features = np.zeros((num_samples, num_features))


    for i in range(num_features):
        fft_result = np.fft.fft(data[:, i])
        magnitude_features[:, i] = np.abs(fft_result)
        phase_features[:, i] = np.angle(fft_result)

    # 这里只取前一半的频域特征
    num_freq_bins = magnitude_features.shape[1] // 2
    magnitude_features = magnitude_features[:, :num_freq_bins]
    phase_features = phase_features[:, :num_freq_bins]

    return np.concatenate([magnitude_features, phase_features], axis=1)

#PCA主成分分析
def pinyu(a):
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)


    # 计算频域特征
    freq_domain_features = compute_fft_features(features_scaled)
    # 进行 PCA 变换
    pca = PCA(n_components=a)  # 或根据需要设置 n_components
    features_pca = pca.fit_transform(freq_domain_features)

    # 转换为 PyTorch 张量
    tensor = torch.from_numpy(features_pca).float()


    return tensor

#没有PCA
# def pinyu(a):
#     # 标准化特征
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)
#
#     # 计算频域特征
#     freq_domain_features = compute_fft_features(features_scaled)
#
#     # 转换为 PyTorch 张量
#     tensor = torch.from_numpy(freq_domain_features).float()
#
#     return tensor

#pinyu()




