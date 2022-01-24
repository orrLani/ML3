import numpy as np

# from paintings import *
# from auxiliary_functions import *

from sklearn.metrics import mean_squared_error

# from sklearn.decomposition import PCA
# from scipy.linalg import hadamard
from sklearn.svm import SVC
# from M import create_bins
from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import minmax_scale
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.isotonic import IsotonicRegression
# from scipy.linalg import tri
from sklearn.preprocessing import StandardScaler

class FeaturePredictingLearner:
    def __init__(self, d):
        self.d = d
        pass

    def fit(self, X, Y):
        # scale = np.var(X, axis=0)
        # assert len(scale) == old_d
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        self.poly = PolynomialFeatures(self.d)
        X = self.poly.fit_transform(X)
        # X[X < 0] = 0
        # X = np.concatenate([X, X_], axis=1)

        self.Ys = list(set(Y))
        X_0 = [x for i, x in enumerate(X) if Y[i] == self.Ys[0]]
        features_0 = np.transpose(X_0)
        X_1 = [x for i, x in enumerate(X) if Y[i] == self.Ys[1]]
        features_1 = np.transpose(X_1)
        self.regressions = []
        self.conf_0 = []
        self.conf_1 = []
        for feature_index, (feature_0, feature_1) in enumerate(zip(features_0, features_1)):
            features_minus_feature_0 = [feature_ for feature_index_, feature_ in enumerate(features_0)
                                        if feature_index_ != feature_index]
            features_minus_feature_1 = [feature_ for feature_index_, feature_ in enumerate(features_1)
                                        if feature_index_ != feature_index]

            X_minus_feature_0 = np.transpose(features_minus_feature_0)
            X_minus_feature_1 = np.transpose(features_minus_feature_1)

            # regression_0 = DecisionTreeRegressor(max_depth=3)
            regression_0 = LinearRegression()
            # regression_0 = DecisionTreeRegressor()
            # regression_0 = KNeighborsRegressor(self.d1)
            regression_0.fit(X_minus_feature_0, feature_0)
            true_score_0 = np.var(regression_0.predict(X_minus_feature_0) - np.array(feature_0))
            # false_score_0 = np.average(abs(regression_0.predict(X_minus_feature_1) - np.array(feature_1)))
            self.conf_0.append(true_score_0)
            # regression_0_score = true_score_0 / false_score_0

            regression_1 = LinearRegression()
            # regression_1 = KernelRidge(kernel='rbf')
            # regression_1 = KNeighborsRegressor(self.d1)

            # regression_1 = DecisionTreeRegressor()

            regression_1.fit(X_minus_feature_1, feature_1)
            # print(new_features)

            true_score_1 = np.var(regression_1.predict(X_minus_feature_1) - np.array(feature_1))
            # false_score_1 = np.average(abs(regression_1.predict(X_minus_feature_0) - np.array(feature_0)))
            self.conf_1.append(true_score_1)

            # regression_1_score = true_score_1 / false_score_1
            self.regressions.append((feature_index, regression_0, regression_1))

        self.conf_0 = np.array(self.conf_0)
        self.conf_0 = self.conf_0
        self.conf_1 = np.array(self.conf_1)
        self.conf_1 = self.conf_1

    def predict_row(self, x):
        expected_x_0 = []
        expected_x_1 = []

        for feature_index, regression_0, regression_1 in self.regressions:
            x_0_minus_feature = np.array([t for i, t in enumerate(x)
                                          if i != feature_index]).reshape(1, -1)
            x_1_minus_feature = np.array([t for i, t in enumerate(x)
                                          if i != feature_index]).reshape(1, -1)

            y0 = regression_0.predict(x_0_minus_feature)[0]
            y1 = regression_1.predict(x_1_minus_feature)[0]
            expected_x_0.append(y0)
            expected_x_1.append(y1)

        diffs_0 = np.sum([np.abs(x[i] - expected_x_0[i]) for i in range(len(x))])
        diffs_1 = np.sum([np.abs(x[i] - expected_x_1[i]) for i in range(len(x))])

        if diffs_1 > diffs_0:
            return self.Ys[0]
        else:
            return self.Ys[1]

    def predict(self, X):
        X = self.scaler.transform(X)
        X = self.poly.transform(X)
        prediction = []
        for x in X:
            prediction.append(self.predict_row(x))
        return prediction

import pandas as pd
from run_code import prepare_data, Q2 , prepare_x_train_y_train
if __name__ == '__main__':
    X, Y = [], []
    # X, Y = blop(X, Y, 0, (0, 0), 0.3, 40)
    # X, Y = blop(X, Y, 0, (1, 1), 0.3, 40)
    # X, Y = blop(X, Y, 1, (0, 1), 0.3, 40)
    # X, Y = blop(X, Y, 1, (1, 0), 0.3, 40)

    # X, Y = blop(X, Y, 0, (0, 0), 0.2, 20)
    # X, Y = blop(X, Y, 1, (0, 1), 0.2, 40)
    # X, Y = blop(X, Y, 0, (0, 2), 0.2, 20)
    # plot_data(X, Y)

    # X, Y = blop(X, Y, 0, (0, 0), 0.3, 40)
    # X, Y = blop(X, Y, 1, (3, 1), 0.3, 40)
    # X, Y = blop(X, Y, 1, (1, 0), 0.3, 40)
    # X, Y = blop(X, Y, 0, (4, 1), 0.3, 40)
    #
    # X, Y = blop(X, Y, 0, (0, 0), 0.3, 40)
    # X, Y = blop(X, Y, 0, (1, 1), 0.3, 40)
    # X, Y = blop(X, Y, 1, (0, 1), 0.3, 40)
    # X, Y = blop(X, Y, 1, (1, 0), 0.3, 40)
    # X, Y = blop(X, Y, 1, (2, 1), 0.3, 40)
    # X, Y = blop(X, Y, 0, (2, 0), 0.3, 40)
    #
    # X, Y = blop(X, Y, 0, (3, 1), 0.3, 40)
    # X, Y = blop(X, Y, 0, (3, 1), 0.3, 40)
    # X, Y = blop(X, Y, 1, (2, 1), 0.3, 40)
    # X, Y = blop(X, Y, 1, (3, 0), 0.3, 40)
    # X, Y = blop(X, Y, 0, (0, 2), 0.3, 40)
    # X, Y = blop(X, Y, 1, (3, 2), 0.3, 40)

    # X, Y = blop(X, Y, 0, (1, 3), 0.3, 40)
    # X, Y = blop(X, Y, 1, (0, 3), 0.3, 40)
    # X, Y = blop(X, Y, 1, (1, 2), 0.3, 40)
    # X, Y = blop(X, Y, 0, (1, 1), 0.3, 20)
    # X, Y = blop(X, Y, 1, (0, 2), 0.2, 20)
    # X, Y = blop(X, Y, 1, (2, 1), 0.3, 20)
    # X, Y = blop(X, Y, 0, (1, 2), 0.3, 20)
    # X, Y = blop(X, Y, 1, (2, 2), 0.3, 20)
    # X, Y = blop(X, Y, 1, (-1, 1), 0.3, 20)
    # X, Y = sphere(X, Y, 0, (0, 0), 1, 50)
    # X, Y = sphere(X, Y, 1, (0, 0), 2, 50)
    # X, Y = sphere(X, Y, 0, (0, 0), 3, 50)
    # X, Y = sphere(X, Y, 1, (0, 0), 4, 50)

    data = pd.read_csv("virus_labeled.csv")
    train_data, test_data = prepare_data(data)
    # Q1(train_data)
    train_data, test_data = Q2(train_data, test_data)

    X, Y = prepare_x_train_y_train(train_data)

    Xtest , Ytest =prepare_x_train_y_train(test_data)
    for d in range(1, 6):
        clf = FeaturePredictingLearner(d=d)
        clf.fit(X, Y)
        res =clf.predict(Xtest)
        # plot_2d_predictions_borders_for_trained_clf(clf, X, Y, f'FeaturePredictingLearner with {d}')
        print("mse ", mean_squared_error(Ytest, res))
        # svc = SVC(kernel='poly', degree=d)
        # svc.fit(X, Y)
        # plot_2d_predictions_borders_for_trained_clf(svc, X, Y, 'SVC with deg ' + str(d))
