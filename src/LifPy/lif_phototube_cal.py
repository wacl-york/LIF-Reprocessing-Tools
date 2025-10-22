import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

cal_data = pd.read_csv('C:\\Users\\pp835\\OneDrive - University of York\\Documents\\Data Analysis\\CARES\\Mace Head Binary Data Analysis\\Phototube Calibration\\PT Cal.txt', delimiter='\t')

# fit a linear regression
X = cal_data['PT / V'].values.astype(np.float64).reshape(-1, 1)
Y = cal_data['Ophir / mW'].values.reshape(-1, 1)
linear_regressor = LinearRegression()
reg = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(cal_data['PT / V'].values.astype(np.float64).reshape(-1, 1))

slope = reg.coef_[0][0]
intercept = reg.intercept_[0]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(X, Y, marker = 'o', linestyle = '')
ax.plot(X, Y_pred, label = 'y = ' + str(slope) + 'x + ' + str(intercept))
ax.legend()
plt.show()