"""Extreme Learning Machine Regression."""
import numpy as np
import sklearn
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score


class RBF(BaseEstimator, RegressorMixin):
    def __init__(self, num_neurons=10, gamma=1, m_weights=None, w_weights=None):
        self.num_neurons = num_neurons
        self.gamma = gamma
        self.w_weights = w_weights  # pesos da camada oculta

    def fit(self, x_train, y_train):
        x_train = np.c_[-1 * np.ones(x_train.shape[0]), x_train]

        kmeans = KMeans(n_clusters=self.num_neurons).fit(x_train)

        self.centers = kmeans.cluster_centers_

        H = rbf_kernel(x_train, self.centers, gamma=self.gamma)

        H = np.c_[-1 * np.ones(H.shape[0]), H]

        # import pdb; pdb.set_trace()

        try:
            self.w_weights = np.linalg.lstsq(H, np.asmatrix(y_train).T, rcond=-1)[0]
        except:
            self.w_weights = np.linalg.pinv(H) @ y_train.reshape(-1, 1)
        return self

    def predict(self, x_test):
        x_test = np.c_[-1 * np.ones(x_test.shape[0]), x_test]

        H = rbf_kernel(x_test, self.centers, gamma=self.gamma)

        H = np.c_[-1 * np.ones(H.shape[0]), H]

        return np.asmatrix(H) @ np.asmatrix(self.w_weights)

    def score(self, X, y, sample_weight=None):
        # from scipy.stats import pearsonr
        # r, p_value = pearsonr(y.reshape(-1, 1), self.predict(X))
        # return r ** 2
        #  Pearson相关系数p的平方 就是判定系数R^2
        return r2_score(y.reshape(-1, 1), self.predict(X))
