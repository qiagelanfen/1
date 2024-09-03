import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('./dataset/traffic/traffic.csv')

# 将第一列转换为日期时间格式
df['Time'] = pd.to_datetime(df.iloc[:, 0])

# 选择第2列到第8列的数据进行可视化
columns_to_plot = df.columns[1:2]  # 第2列到第8列

# 绘制折线图
plt.figure(figsize=(12, 6))

for column in columns_to_plot:
    plt.plot(df['Time'], df[column], label=column)

# 添加图表标题和轴标签
plt.title('Time Series Visualization (Columns 2 to 8)')
plt.xlabel('Time')
plt.ylabel('Values')

# 显示图例
plt.legend()

# 显示图表
plt.show()
