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

additional_file_1 = r"C:\Users\dell\Desktop\主权评级.xlsx"
additional_file_2 = r"C:\Users\dell\Desktop\名义有效汇率指数变动.xlsx"
rank = pd.read_excel(additional_file_1, index_col=0)
delta_exchange = pd.read_excel(additional_file_2, index_col=0)
delta_exchange[delta_exchange < 0] = 0
# 这个mapping有一定的任意性，具体原因见论文正文
mapping = {'AA-': 1, 'A+': 2, 'A': 3, 'A-': 4, 'BBB+': 5, 'BBB': 6, 'BBB-': 7, 'BB+': 8, 'BB': 9, 'BB-': 10,
           'B+': 11, 'B': 12, 'B-': 13, 'CCC+': 14, 'CCC': 15, 'CCC-': 16, 'CC+': 17, 'CC': 18}
for i in mapping.keys():
    rank[rank == i] = mapping[i]

data_new = pd.concat([data, delta_exchange], axis=1)
for col in rank.columns:
    data_new[col] = 0
for ind in data_new.index:
    for col in rank.columns:
        data_new.loc[ind, col] = rank.loc[ind // 100, col]
country_list = ['Peru', 'India']
column_list = list(map(lambda x: 'G\'(x)_estimation_' + str(x), country_list)) + \
              list(map(lambda x: 'weight_' + str(x), country_list)) + \
              list(map(lambda x: str(x) + '_Rpre', country_list)) + \
              list(map(lambda x: str(x) + '_rk', country_list))

data_for_model_training = data_new.loc[:, column_list].dropna(how='any')

# 这一小段的作图是为了展示报告的直观性，与实际回归有所差别
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
data1 = pd.DataFrame(data_for_model_training.loc[:, ['weight_Peru', 'Peru_Rpre', 'Peru_rk']])
data1.columns = [1, 2, 3]
data2 = pd.DataFrame(data_for_model_training.loc[:, ['weight_India', 'India_Rpre', 'India_rk']])
data2.columns = [1, 2, 3]
data_for_regression_x = pd.concat([data1, data2], ignore_index=True)
data3 = pd.DataFrame(data_for_model_training.loc[:, 'G\'(x)_estimation_Peru'])
data3.columns = ['y']
data4 = pd.DataFrame(data_for_model_training.loc[:, 'G\'(x)_estimation_India'])
data4.columns = ['y']
data_for_regression_y = pd.concat([data3, data4], ignore_index=True)

# adfvalues1 = adfuller(data_for_regression_x, 1)
# adfvalues2 = adfuller(data_for_regression_y, 1)
data_for_regression_x_add = sms.add_constant(data_for_regression_x)
model = sms.OLS(data_for_regression_y, data_for_regression_x_add).fit()
summary = model.summary()
print(summary)
# print('adfuller1=', adfvalues1)
# print('adfuller2=', adfvalues2)

# fig_1 = plt.figure(figsize=(8, 4))
# plt.plot([0, 1], [model.predict(exog=(1, 0)), model.predict(exog=(1, 1))], c='red')
# plt.scatter(data_for_regression_x, data_for_regression_y, c='b')
# plt.xlabel('w')
# plt.ylabel('g\'(w)')
# plt.show()

'''
回归的R^2似乎太小，考虑到可能是某一个w附近有太多散度很大的y，下面进行分段平均化再试一次
'''
x_mean = np.linspace(0.025, 0.975, 20)
x_range_data = pd.DataFrame(index=range(20), columns=[1, 2, 3])
y_range_data = pd.DataFrame(index=range(20), columns=['y'])
data_for_regression_x_1 = data_for_regression_x.loc[:,1]
for x in x_mean:
    x_lower = x - 0.025
    x_upper = x + 0.025
    a = np.where(data_for_regression_x_1 > x_lower)
    b = np.where(data_for_regression_x_1 < x_upper)
    condition = np.intersect1d(a, b)
    x_range_data.loc[x//0.024, :] = data_for_regression_x.loc[condition, :].mean()
    y_range_data.loc[x//0.024, :] = data_for_regression_y.loc[condition, :].mean()

data_for_regression_x_range_add = sms.add_constant(x_range_data.dropna())
model = sms.OLS(y_range_data.dropna().astype(float), data_for_regression_x_range_add.astype(float)).fit()
summary = model.summary()
print(summary)

fig_1 = plt.figure(figsize=(8, 4))
# plt.plot([0, 1], [model.predict(exog=(1, 0)), model.predict(exog=(1, 1))], c='red')
# plt.scatter(x_range_data, y_range_data, c='b')
# plt.xlabel('w')
# plt.ylabel('g\'(w)')
# plt.show()

with open('summary.pkl', 'wb') as fl1:
    pickle.dump(summary, fl1)

with open('model.pkl', 'wb') as fl2:
    pickle.dump(model, fl2)

with open('data_after_step_4.pkl', 'wb') as fl3:
    pickle.dump(data_new, fl3)
