# coding: UTF-8

# 使用简单线性回归
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# 读取数据，dataframe格式
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

# 使用sklearn的线性回归库
bmi_life_model = linear_model.LinearRegression()

# 使用fit函数训练数据，注意fit(feature,target)的输入要求是二维array
# [n_samples,n_features]
# [n_samples, n_targets]
# bmi_life_data[['BMI']]返回df df.values是二维数组
bmi_life_model.fit(bmi_life_data[['BMI']],bmi_life_data[['Life expectancy']])

# 使用predict函数预测 fit的输入要求是二维array
# [n_samples, n_features]
laos_life_exp = bmi_life_model.predict([[21.07931]])
print(laos_life_exp)
