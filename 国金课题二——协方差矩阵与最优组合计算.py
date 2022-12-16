import pickle

with open('prepared_data.pkl', 'rb') as fl:
    data = pickle.load(fl)

import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt

data = data.drop(data.columns[:6], axis=1)
for i in range(1, 6):
    # 这里j从1开始而不是从i开始是为了方便下面构建协方差矩阵
    for j in range(1, 6):
        column_name = 'cov_' + str(i) + '_' + str(j)
        data[column_name] = data.iloc[:, i - 1].rolling(60).cov(data.iloc[:, j - 1])

for i in range(5):
    data.iloc[:, i] = data.iloc[:, i].rolling(60).mean()
data = data.dropna(how='any', axis=0)

file_name = r"C:\Users\dell\Desktop\美国_国债收益率_10年_月_平均值.xlsx"
risk_free_rate_data = pd.read_excel(file_name, index_col=0) / 100


def statistic(weights, return_series, covariance_matrix, risk_free_rate):
    weights = np.array(weights)
    port_return = np.dot(weights, return_series.T) * 12
    port_var = np.dot(np.dot(weights, covariance_matrix), weights.T) * 12
    sharpe_ratio = (port_return - risk_free_rate) / np.sqrt(port_var)
    return sharpe_ratio, port_return, np.sqrt(port_var)


for index in range(len(data.index)):
    return_series = np.array(data.iloc[index, :5])
    covariance_matrix = np.array([data.iloc[index, 5:10],
                                  data.iloc[index, 10:15],
                                  data.iloc[index, 15:20],
                                  data.iloc[index, 20:25],
                                  data.iloc[index, 25:30]])
    risk_free_rate = risk_free_rate_data.loc[data.index[index], 'rate']


    def sharpe(weights):
        return -statistic(weights, return_series, covariance_matrix, risk_free_rate)[0]


    constraint = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple(((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)))
    opts_1 = sco.minimize(sharpe, 5 * [1. / 5, ], method='SLSQP', bounds=None, constraints=constraint)
    opts_2 = sco.minimize(sharpe, 5 * [1. / 5, ], method='SLSQP', bounds=bnds, constraints=constraint)

    port_return, port_std = statistic(opts_1['x'], return_series, covariance_matrix, risk_free_rate)[1:]
    port_return_1, port_std_1 = statistic(opts_2['x'], return_series, covariance_matrix, risk_free_rate)[1:]
    data.loc[data.index[index], 'port_return_with_short'] = port_return
    data.loc[data.index[index], 'port_std_with_short'] = port_std
    data.loc[data.index[index], 'port_return_without_short'] = port_return_1
    data.loc[data.index[index], 'port_std_without_short'] = port_std_1
with open('optimal_port_data.pkl', 'wb') as fl:
    pickle.dump(data, fl)

'''
以下部分并非必要，只是为了在报告中直观展示上面计算的结果而取一个时刻作图
我们随意取一个月份展示如何求出世界最有资产组合的收益率和标准差
'''

return_series = np.array(data.iloc[-1, :5])
covariance_matrix = np.array([data.iloc[-1, 5:10],
                              data.iloc[-1, 10:15],
                              data.iloc[-1, 15:20],
                              data.iloc[-1, 20:25],
                              data.iloc[-1, 25:30]])
risk_free_rate = risk_free_rate_data.loc[data.index[-1], 'rate']

port_return = []
port_std = []
for i in range(40000):
    weights = np.random.uniform(size=5)
    weights /= np.sum(weights)
    port_return.append(statistic(weights, return_series, covariance_matrix, risk_free_rate)[1])
    port_std.append(statistic(weights, return_series, covariance_matrix, risk_free_rate)[2])

plt.figure(figsize=(8, 4))
plt.scatter(port_std, port_return, c=(port_return - risk_free_rate) / port_std, marker='o')
plt.grid(True)
plt.xlabel('excepted std')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.plot((0, data.iloc[-1, -1] * 2), (risk_free_rate, data.iloc[-1, -2] * 2 - risk_free_rate))
plt.xlim([0, 0.2])
plt.ylim([-0, 0.2])
plt.show()
'''
上面在画不允许卖空的图像，下面是允许卖空的图像，方法是一模一样的，因此读者可以不详细阅读下段代码
'''
port_return = []
port_std = []
for i in range(40000):
    weights = np.random.uniform(size=5) - 0.5
    weights /= np.sum(weights)
    port_return.append(statistic(weights, return_series, covariance_matrix, risk_free_rate)[1])
    port_std.append(statistic(weights, return_series, covariance_matrix, risk_free_rate)[2])

plt.figure(figsize=(8, 4))
plt.scatter(port_std, port_return, c=(port_return - risk_free_rate) / port_std, marker='o')
plt.grid(True)
plt.xlabel('excepted std')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.plot((0, data.iloc[-1, -3] * 2), (risk_free_rate, data.iloc[-1, -4] * 2 - risk_free_rate))
plt.xlim([0, 1])
plt.ylim([-0.5, 1])
plt.show()
