import pickle

with open('after_step_3_data.pkl', 'rb') as fl:
    data = pickle.load(fl)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sms
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

filename = r"C:\Users\dell\Desktop\IIP_ljq_G函数所需.xlsx"
weight_data = pd.read_excel(filename, index_col=0)

file_name_additional = r"C:\Users\dell\Desktop\IIP补充数据.xlsx"
weight_data_additional = pd.read_excel(file_name_additional, index_col=0)

file_name = r"C:\Users\dell\Desktop\美国_国债收益率_10年_月_平均值.xlsx"
risk_free_rate_data = pd.read_excel(file_name, index_col=0) / 100

for i in range(4):
    weight_data.iloc[:, i * 3 + 2] = 1 - weight_data.iloc[:, i * 3 + 1] / weight_data.iloc[:, i * 3]
    weight_data_additional.iloc[:, i * 3 + 2] = 1 - weight_data_additional.iloc[:,
                                                    i * 3 + 1] / weight_data_additional.iloc[:, i * 3]

sample_countries_name = ['Bahrain', 'India', 'Denmark']
sample_countries_name_additional = ['Peru', 'Maldives', 'Ukraine', 'Argentina']

for country in sample_countries_name_additional:
    data['G\'(x)_estimation_' + country] = 0
for country in sample_countries_name:
    data['G\'(x)_estimation_' + country] = 0

for index in data.index:
    for column in [-3, -2, -1]:
        if column == -1:
            gamma = -8
        else:
            gamma = -9
        data.loc[index, data.columns[column]] = 0.5 * data.loc[index, data.columns[gamma]] * weight_data.loc[
            index // 100, weight_data.columns[11 + 3 * column]] * data.loc[index, 'port_std_without_short'] ** 2 - (
                                                        data.loc[index, 'port_return_without_short'] -
                                                        risk_free_rate_data.loc[index, 'rate'])
    for column in [-7, -6, -5, -4]:
        data.loc[index, data.columns[column]] = 0.5 * data.loc[index, data.columns[-9]] * weight_data_additional.loc[
            index // 100, weight_data_additional.columns[23 + 3 * column]] * data.loc[
                                                    index, 'port_std_without_short'] ** 2 - (
                                                        data.loc[index, 'port_return_without_short'] -
                                                        risk_free_rate_data.loc[index, 'rate'])

for country in sample_countries_name_additional:
    data['weight_' + country] = 0
for country in sample_countries_name:
    data['weight_' + country] = 0

for index in data.index:
    for column in [-3, -2, -1]:
        data.loc[index, data.columns[column]] = weight_data.loc[index // 100, weight_data.columns[11 + 3 * column]]
    for column in [-7, -6, -5, -4]:
        data.loc[index, data.columns[column]] = weight_data_additional.loc[
            index // 100, weight_data_additional.columns[23 + 3 * column]]

for i in range(-14, -7, -1):
    data.iloc[:, i] = data.iloc[:, i].rolling(12).mean()

fig = plt.figure(figsize=(8, 4))
for country_index in [1, 4, 6]:
    plt.scatter(data.iloc[11::12, -country_index - 1], data.iloc[11::12, -country_index - 8],
                color=(country_index / 6, 0, 1 - country_index / 6))
sample_countries_name_additional.extend(sample_countries_name)
plt.legend(['India', 'Ukraine', 'Peru'])
plt.xlabel('w')
plt.ylabel('g\'(w)')
plt.show()

'''
这里的[1,4,6]是怎么来的呢？最开始是画range(7)，七个国家都画，然后根据散点图把x散度太小的四个国家剔除出去
'''

data_for_regression_x = np.array([data.iloc[11::12, -country_index - 1] for country_index in [1, 4, 6]],
                                 dtype=np.float32).flatten()
data_for_regression_y = np.array([data.iloc[11::12, -country_index - 8] for country_index in [1, 4, 6]],
                                 dtype=np.float32).flatten()
data_for_regression_x = np.delete(data_for_regression_x, np.where(np.isnan(data_for_regression_x)))
data_for_regression_y = np.delete(data_for_regression_y, np.where(np.isnan(data_for_regression_y)))
data_for_regression_x = data_for_regression_x.reshape(-1, 1)
data_for_regression_y = data_for_regression_y.reshape(-1, 1)

adfvalues1 = adfuller(data_for_regression_x, 1)
adfvalues2 = adfuller(data_for_regression_y, 1)
data_for_regression_x_add = sms.add_constant(data_for_regression_x)
model = sms.OLS(data_for_regression_y, data_for_regression_x_add).fit()
summary = model.summary()
print(summary)
print('adfuller1=', adfvalues1)
print('adfuller2=', adfvalues2)

fig_1 = plt.figure(figsize=(8, 4))
plt.plot([0, 1], [model.predict(exog=(1, 0)), model.predict(exog=(1, 1))], c='red')
plt.scatter(data_for_regression_x, data_for_regression_y, c='b')
plt.xlabel('w')
plt.ylabel('g\'(w)')
plt.show()

'''
回归的R^2似乎太小，考虑到可能是某一个x附近有太多散度很大的y，下面进行分段平均化再试一次
'''
x_mean = np.linspace(0.025, 0.975, 20)
x_range_data = []
y_range_data = []
data_for_regression_x = data_for_regression_x.flatten()
data_for_regression_y = data_for_regression_y.flatten()
for x in x_mean:
    x_lower = x - 0.025
    x_upper = x + 0.025
    a = np.where(data_for_regression_x > x_lower)
    b = np.where(data_for_regression_x < x_upper)
    condition = np.intersect1d(a, b)
    x_range_data.append(data_for_regression_x[condition].mean())
    y_range_data.append(data_for_regression_y[condition].mean())

x_range_data = np.delete(x_range_data, np.where(np.isnan(x_range_data)))
y_range_data = np.delete(y_range_data, np.where(np.isnan(y_range_data)))
x_range_data = np.array(x_range_data).reshape(-1, 1)
y_range_data = np.array(y_range_data).reshape(-1, 1)

data_for_regression_x_range_add = sms.add_constant(x_range_data)
model = sms.OLS(y_range_data, data_for_regression_x_range_add).fit()
summary = model.summary()
print(summary)

fig_1 = plt.figure(figsize=(8, 4))
plt.plot([0, 1], [model.predict(exog=(1, 0)), model.predict(exog=(1, 1))], c='red')
plt.scatter(x_range_data, y_range_data, c='b')
plt.xlabel('w')
plt.ylabel('g\'(w)')
plt.show()

with open('summary.pkl', 'wb') as fl1:
    pickle.dump(summary, fl1)

with open('model.pkl', 'wb') as fl2:
    pickle.dump(model, fl2)

with open('data_after_step_4.pkl', 'wb') as fl3:
    pickle.dump(data, fl3)