import numpy as np
from rbf_regressor import RBF

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

params = {'num_neurons': list(np.linspace(10, 100, 10).astype(int)),
          "gamma": list(np.linspace(0.01, 1, 10))}

boston = load_boston()

# (506, 13)
# print(boston.data.shape)

scaler = MinMaxScaler().fit(boston.data)

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# model = RBF(num_neurons=20, gamma=2**5)

# model.fit(X_train, y_train)

params = {'gamma': [0.34], 'num_neurons': [100]}

gs = GridSearchCV(RBF(), params, cv=5, error_score='raise')
gs.fit(X_train, y_train)
model = gs.best_estimator_

y_hat = model.predict(X_test)

'''
mean_squared_error(y_test, y_hat) 得到的结果是均方误差
mean_squared_error(y_test, y_hat) ** 0.5 得到的结果是均方根误差
'''
print('均方根误差：{}   R**2：{}'.format(mean_squared_error(y_test, y_hat) ** .5, model.score(X_test, y_test)))
print(gs.best_params_)
