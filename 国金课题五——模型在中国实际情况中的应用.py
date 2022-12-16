import pickle
import statsmodels.api as sms  # 这个引用是必要的，尽管编译器会显示这个引用没有用到。这是因为下面model.pkl里面封装的是sms对象

with open('model.pkl', 'rb') as fl:
    model = pickle.load(fl)

with open('data_after_step_4.pkl', 'rb') as fl1:
    data = pickle.load(fl1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = r"C:\Users\dell\Desktop\IIP_ljq_G函数所需.xlsx"
China_data = pd.read_excel(filename, index_col=0)

China_data.iloc[:, -1] = 1 - China_data.iloc[:, -2] / China_data.iloc[:, -3]
China_weight = China_data.iloc[:, -1].dropna()
data = data.loc['200401':, :]
data['China_weight'] = 0
for date in data.index:
    data.loc[date, 'China_weight'] = China_weight[date // 100]

filename1 = r"C:\Users\dell\Desktop\美国_国债收益率_10年_月_平均值.xlsx"
risk_free_rate = pd.read_excel(filename1, index_col=0)

data['China_U\'(x)_as_developing_country'] = 0
data['China_U\'(x)_as_developed_country'] = 0
for index in data.index:
    # 又是一个不用户友好的公式，关于这个公式还是详见论文正文
    data.loc[index, 'China_U\'(x)_as_developed_country'] = -(
            data.loc[index, 'port_return_without_short'] - risk_free_rate.loc[
        index, 'rate'] - 1 / 2 * data.loc[index, 'developed_countries_risk_averse_factor_estimation'] * data.loc[
                index, 'China_weight'] * data.loc[
                index, 'port_std_without_short'] ** 2 + model.predict(
        (1, data.loc[index, 'China_weight'])))
    data.loc[index, 'China_U\'(x)_as_developing_country'] = -(
            data.loc[index, 'port_return_without_short'] - risk_free_rate.loc[
        index, 'rate'] - 1 / 2 * data.loc[index, 'developing_countries_risk_averse_factor_estimation'] * data.loc[
                index, 'China_weight'] * data.loc[
                index, 'port_std_without_short'] ** 2 + model.predict(
        (1, data.loc[index, 'China_weight'])))
data.index = data.index // 100 + (data.index % 100) / 12
fig = plt.figure(figsize=(8, 4))
plt.plot(data.iloc[:, -2], color='b')
plt.plot(data.iloc[:, -1], color='g')
plt.legend(data.columns[-2:])
plt.show()
