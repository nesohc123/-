import pickle

with open('optimal_port_data.pkl', 'rb') as fl:
    data = pickle.load(fl)

import pandas as pd
import matplotlib.pyplot as plt

'''
接下来的分析不允许卖空国外资产，原因有二：
第一是实际中投资者卖空国外资产有一定难度，且不是自由卖空的
第二是有一些年份，某几个国家过去五年数据高度共线性（似乎可以套利），允许卖空会出现非常荒谬的结果
'''

filename = r"C:\Users\dell\Desktop\IIP数据_dqw_风险厌恶系数所需.xlsx"
weight_data = pd.read_excel(filename, index_col=0)
for i in range(7):
    weight_data.iloc[:, i * 3 + 2] = 1 - weight_data.iloc[:, i * 3 + 1] / weight_data.iloc[:, i * 3]

sample_countries_name = ['Korea', 'Poland', 'Indonesia', 'Malaysia', 'England', 'Germany', 'Australia']

fig1 = plt.figure(figsize=(8, 4))
for i in range(7):
    plt.plot(weight_data.iloc[:, i * 3 + 2], c=((150 + 10 * i) / 255, (200 - 10 * i) / 255, (50 + 15 * i) / 255))
plt.legend(sample_countries_name)
plt.show()

file_name = r"C:\Users\dell\Desktop\美国_国债收益率_10年_月_平均值.xlsx"
risk_free_rate_data = pd.read_excel(file_name, index_col=0) / 100

for country in sample_countries_name:
    data['risk_averse_factor_estimation_' + country] = 0
for index in range(len(data.index)):
    for i in range(7):
        # 下面这个公式确实不是很用户友善，为什么是这样还请参考论文解释
        data.iloc[index, i - 7] = (data.iloc[index, -9] - risk_free_rate_data.loc[data.index[index], 'rate']) / (
                    2 * weight_data.loc[data.index[index] // 100, weight_data.columns[3 * i + 2]] * data.iloc[
                index, -8] ** 2)

data.index = data.index // 100 + (data.index % 100) / 12
fig = plt.figure(figsize=(8, 4))
for i in range(4):
    plt.plot(data.iloc[:, i - 7], c=((150 + 10 * i) / 255, (200 - 10 * i) / 255, (50 + 15 * i) / 2550))
plt.legend(sample_countries_name[:4])
plt.show()

data['developing_countries_risk_averse_factor_estimation'] = data.iloc[:, [-7, -6, -5, -4]].mean(axis=1)
data['developed_countries_risk_averse_factor_estimation'] = data.iloc[:, [-3, -2, -1]].mean(axis=1)

with open('after_step_3_data.pkl', 'wb') as fl:
    pickle.dump(data, fl)
