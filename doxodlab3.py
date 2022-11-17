import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df = pd.DataFrame({
    'Месяц\год': ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'],

    '2017': [65.000, 61.000, 63.000, 69.000, 70.580, 97.365, 104.755, 101.820, 83.655, 77.910, 70.365, 64.200],
    '2018': [69.550, 65.270, 67.410, 73.830, 75.521, 104.181, 112.088, 108.947, 89.511, 83.364, 75.291, 68.694],
    '2019': [71.358, 66.967, 69.163, 75.750, 77.484, 106.889, 115.002, 111.780, 91.838, 85.531, 77.248, 70.480],
    '2020': [77.781, 72.994, 75.387, 82.567, 84.458, 116.509, 125.352, 121.840, 100.104, 93.229, 84.200, 76.823],
    '2021': [81.670, 76.644, 79.157, 86.695, 88.681, 122.335, 131.620, 127.932, 105.109, 97.890, 88.410, 80.664],
    '2022': [89.837, 84.308, 87.072, 95.365, 97.549, 134.568, 144.782, 140.725, 115.620, 107.679, 97.252, 88.731],
    '2023': [97.832, 91.812, 94.822, 103.852, 106.230, 146.545, 157.668, 153.250, 125.910, 117.263, 105.907, 96.628],
    '2024': [106.735, 100.166, 103.451, 113.303, 115.897, 159.880, 172.015, 167.196, 137.368, 127.934, 115.544, 105.421]})


pd.set_option('display.max_columns', None)
df.head()
print(df)


print('Оптимистичный прогноз:')
d = {'2023': pd.Series([111.305, 105.285, 108.295, 117.326, 119.704, 160.018, 171.141, 166.723, 139.383, 130.736, 119.380, 110.101],
                            index = ['январь', 'фeвраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']),
           '2024': pd.Series([121.434, 114.866, 118.150, 128.002, 130.597, 174.580, 186.715, 181.895, 152.067, 142.633, 130.244, 120.120],
                            index = ['январь', 'фeвраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'])}
df = pd.DataFrame(d)
print(df)


print('Пессимистичный прогноз:')

d = {'2023': pd.Series([84.359, 78.338, 81.349, 90.379, 92.757, 133.072, 144.194, 139.777, 112.437, 103.790, 92.434, 83.155],
                            index = ['январь', 'фeвраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']),
           '2024': pd.Series([92.035, 85.467, 88.751, 98.604, 101.198, 145.181, 157.316, 152.497, 122.668, 113.235, 100.845, 90.722],
                            index = ['январь', 'фeвраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'])}
df = pd.DataFrame(d)
print(df)




#Winter2023
x = np.array([1, 2, 3]).reshape((-1, 1))
y = np.array([96.628, 97.832, 91.812])
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept:', new_model.intercept_)
print('slope:', new_model.coef_)

y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')


#Spring2023
x = np.array([4, 5, 6]).reshape((-1, 1))
y = np.array([94.822, 103.852, 106.230])
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination Spring2023:', r_sq)
print('intercept Spring2023:', model.intercept_)
print('slope Spring2023:', model.coef_)
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept Spring2023:', new_model.intercept_)
print('slope Spring2023:', new_model.coef_)

y_pred = model.predict(x)
print('predicted response Spring2023:', y_pred, sep='\n')


#Summer2023
x = np.array([7, 8, 9]).reshape((-1, 1))
y = np.array([146.545, 157.668, 153.250])
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination Summer2023:', r_sq)
print('intercept Summer2023:', model.intercept_)
print('slope Summer2023:', model.coef_)
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept Summer2023:', new_model.intercept_)
print('slope Summer2023:', new_model.coef_)

y_pred = model.predict(x)
print('predicted response Summer2023:', y_pred, sep='\n')


#Autumn2023
x = np.array([10, 11, 12]).reshape((-1, 1))
y = np.array([125.910, 117.263, 105.907])
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination Autumn2023:', r_sq)
print('intercept Autumn2023:', model.intercept_)
print('slope Autumn2023:', model.coef_)
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept Autumn2023:', new_model.intercept_)
print('slope Autumn2023:', new_model.coef_)

y_pred = model.predict(x)
print('predicted response Autumn2023:', y_pred, sep='\n')



#Winter2024
x = np.array([1, 2, 3]).reshape((-1, 1))
y = np.array([105.421, 106.735, 100.166])
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept:', new_model.intercept_)
print('slope:', new_model.coef_)

y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')


#Spring2024
x = np.array([4, 5, 6]).reshape((-1, 1))
y = np.array([103.451, 113.303, 115.897])
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination Spring2024:', r_sq)
print('intercept Spring2024:', model.intercept_)
print('slope Spring2024:', model.coef_)
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept Spring2024:', new_model.intercept_)
print('slope Spring2024:', new_model.coef_)

y_pred = model.predict(x)
print('predicted response Spring2024:', y_pred, sep='\n')


#Summer2024
x = np.array([7, 8, 9]).reshape((-1, 1))
y = np.array([159.880, 172.015, 167.196])
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination Summer2024:', r_sq)
print('intercept Summer2024:', model.intercept_)
print('slope Summer2024:', model.coef_)
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept Summer2024:', new_model.intercept_)
print('slope Summer2024:', new_model.coef_)

y_pred = model.predict(x)
print('predicted response Summer2024:', y_pred, sep='\n')


#Autumn2024
x = np.array([10, 11, 12]).reshape((-1, 1))
y = np.array([137.368, 127.934, 115.544])
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination Autumn2023:', r_sq)
print('intercept Autumn2024:', model.intercept_)
print('slope Autumn2024:', model.coef_)
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept Autumn2024:', new_model.intercept_)
print('slope Autumn2024:', new_model.coef_)

y_pred = model.predict(x)
print('predicted response Autumn2024:', y_pred, sep='\n')






