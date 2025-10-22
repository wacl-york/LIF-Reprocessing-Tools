import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# cell A flow calibration
sig_A_flow_cal = {
    'set_flow_slpm' : [1, 2, 3, 4, 5],
    'repeat_1' : [1.1607, 2.2277, 3.2991, 4.3700, 5.4171],
    'repeat_2' : [1.1645, 2.2320, 3.3038, 4.3847, 5.4275],
    'repeat_3' : [1.1480, 2.2371, 3.3058, 4.3701, 5.4288]
}
sig_A_flow_cal = pd.DataFrame(sig_A_flow_cal)
sig_A_flow_cal['avg_flow'] = sig_A_flow_cal[['repeat_1', 'repeat_2', 'repeat_3']].mean(axis=1)

X = sig_A_flow_cal['set_flow_slpm'].values.reshape(-1, 1)
Y = sig_A_flow_cal['avg_flow'].values.reshape(-1, 1)
linear_regressor = LinearRegression()
reg = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)
flow_vars = {}
flow_dict = {}
flow_dict['R2'] = reg.score(X,Y)
flow_dict['Slope'] = reg.coef_[0,0]
flow_dict['Intercept'] = reg.intercept_[0]
flow_vars['sig_A'] = flow_dict

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.title('sig_A_flow_cal')
plt.show()


# cell B flow calibration
sig_B_flow_cal = {
    'set_flow_slpm' : [1, 2, 3, 4, 5],
    'repeat_1' : [1.0894, 2.1533, 3.1936, 4.2193, 5.2643],
    'repeat_2' : [1.0960, 2.1563, 3.1804, 4.2174, 5.2627],
    'repeat_3' : [1.0946, 2.1618, 3.1892, 4.2180, 5.2637]
}
sig_B_flow_cal = pd.DataFrame(sig_B_flow_cal)
sig_B_flow_cal['avg_flow'] = sig_B_flow_cal[['repeat_1', 'repeat_2', 'repeat_3']].mean(axis=1)

X = sig_B_flow_cal['set_flow_slpm'].values.reshape(-1, 1)
Y = sig_B_flow_cal['avg_flow'].values.reshape(-1, 1)
linear_regressor = LinearRegression()
reg = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)
flow_dict = {}
flow_dict['R2'] = reg.score(X,Y)
flow_dict['Slope'] = reg.coef_[0,0]
flow_dict['Intercept'] = reg.intercept_[0]
flow_vars['sig_B'] = flow_dict

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.title('sig_B_flow_cal')
plt.show()


# Ref cell flow calibration
ref_flow_cal = {
    'set_flow_slpm' : [1, 2, 3, 4, 5],
    'repeat_1' : [1.1330, 2.2058, 3.2337, 4.2707, 5.3178],
    'repeat_2' : [1.1285, 2.2008, 3.2363, 4.2691, 5.3151],
    'repeat_3' : [1.1335, 2.2035, 3.2327, 4.2647, 5.3191]
}
ref_flow_cal = pd.DataFrame(ref_flow_cal)
ref_flow_cal['avg_flow'] = ref_flow_cal[['repeat_1', 'repeat_2', 'repeat_3']].mean(axis=1)

X = ref_flow_cal['set_flow_slpm'].values.reshape(-1, 1)
Y = ref_flow_cal['avg_flow'].values.reshape(-1, 1)
linear_regressor = LinearRegression()
reg = linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)
flow_dict = {}
flow_dict['R2'] = reg.score(X,Y)
flow_dict['Slope'] = reg.coef_[0,0]
flow_dict['Intercept'] = reg.intercept_[0]
flow_vars['ref'] = flow_dict

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.title('ref_flow_cal')
plt.show()


print(flow_vars)