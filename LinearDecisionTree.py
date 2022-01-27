from sklearn.svm import LinearSVC, SVC
import numpy as np
# from paintings import *
# from auxiliary_functions import plot_2d_predictions_borders_for_trained_clf
from collections import Counter
from catboost import CatBoostRegressor


class Node:
    def __init__(self, clf, l=None, r=None):
        self.clf = clf
        self.l = l
        self.r = r

class ClassClassifier:
    def __init__(self, c):
        self.c = c

    def predict(self, X):
        return np.array([self.c for _ in range(len(X))])

class LinearDecisionTree:
    def __init__(self, max_depth, svc_C):
        self.max_depth = max_depth
        self.root = None
        self.most_common_y = None
        self.max_depth_visited = -1
        self.svc_C = svc_C

    def get_clf(self, X, Y):
        assert len(X) == len(Y)
        if len(X) == 0:
            print("len(X) == 0")
            return ClassClassifier(self.most_common_y)
        elif len(set(Y)) == 1:
            print("len(set(Y)) == 1")
            return ClassClassifier(Y[0])
        else:
            clf = LinearSVC(C=self.svc_C)
            clf.fit(X, Y)
            return clf

    def build_tree(self, X, Y, depth=0):
        self.max_depth_visited = max(depth, self.max_depth_visited)
        clf = self.get_clf(X, Y)
        # Y_hat = clf.predict(X)
        # acc = np.sum(Y == Y_hat) / len(Y)
        # print(acc)
        if depth == self.max_depth or isinstance(clf, ClassClassifier):
            return Node(clf)
        else:
            values = clf.decision_function(X)
            node = Node(clf)
            node.l = self.build_tree(X[values < 0], Y[values < 0], depth+1)
            node.r = self.build_tree(X[values >= 0], Y[values >= 0], depth+1)
        return node

    def fit(self, X, Y):
        self.most_common_y = Counter(Y).most_common(1)[0][0]
        self.root = self.build_tree(X, Y)
        print(f"max depth visited: {self.max_depth_visited}")

    def predict_x(self, x, n=None):
        if n is None:
            n = self.root
        if n.l is None:
            return n.clf.predict([x])[0]
        else:
            v = n.clf.decision_function([x])[0]
            if v < 0:
                return self.predict_x(x, n.l)
            else:
                return self.predict_x(x, n.r)

    def predict(self, X):
        return [self.predict_x(x) for x in X]


import pandas as pd
from run_code import prepare_data, Q2 , prepare_x_train_y_train
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
if __name__ == "__main__":
    # X, Y = [], []
    # X, Y = function_plot(X, Y, lambda x: -x**2,     -1, (0, 0), 0.3, -1, 2, 60)
    # X, Y = function_plot(X, Y, lambda x: x,         -1, (0, 3), 0.3, -1, 1, 100)
    # X, Y = function_plot(X, Y, lambda x: x**2,       1, (0, 1), 0.3, -1, 1, 100)
    # X, Y = function_plot(X, Y, lambda x: -4 * x,     1, (2, 1), 0.3, -1, 0.5, 100)
    # #
    # X, Y = function_plot(X, Y, lambda x: -x ** 2,    0, (5, 0), 0.3, -1, 2, 60)
    # X, Y = function_plot(X, Y, lambda x: x,          0, (5, 3), 0.3, -1, 1, 90)
    # X, Y = function_plot(X, Y, lambda x: x ** 2,     1, (5, 1), 0.3, -1, 1, 60)
    # X, Y = function_plot(X, Y, lambda x: -4 * x,     0, (7, 1), 0.3, -1, 0.5, 100)
    # X, Y = sphere(X, Y, 0, (0, 0), 1, 50, 1)
    # X, Y = sphere(X, Y, 1, (0, 0), 2, 50, 1)
    # X, Y = sphere(X, Y, 0, (0, 0), 3, 50, 1)
    # X, Y = sphere(X, Y, 1, (0, 0), 4, 50, 1)

    # X, Y = blop(X, Y, 0, (0, 1), 0.5, 80)
    # X, Y = blop(X, Y, 0, (3, 1), 0.5, 80)
    # X, Y = blop(X, Y, 1, (2, 0.5), 0.4, 70)
    # X, Y = blop(X, Y, 1, (1, 0), 0.4, 90)
    # X, Y = blop(X, Y, 1, (4, 1), 0.4, 200)
    # X, Y = blop(X, Y, 0, (2, 2), 0.3, 50)
    # X, Y = blop(X, Y, 1, (1, 2), 0.3, 140)
    # X, Y = blop(X, Y, 0, (3, 3), 0.3, 30)
    # X, Y = blop(X, Y, 1, (3, 2), 0.3, 60)
    # X, Y = blop(X, Y, 1, (2, 3), 0.3, 50)
    # X, Y = blop(X, Y, 0, (1, -1), 0.3, 130)
    # X, Y = blop(X, Y, 1, (4, 0), 0.3, 20)
    # X, Y = blop(X, Y, 0, (2, 1), 0.3, 60)
    # X, Y = blop(X, Y, 1, (1, 3), 0.3, 50)

    # X = np.array(X)
    # Y = np.array(Y)

    data = pd.read_csv("virus_labeled.csv")
    unlabled = pd.read_csv("virus_unlabeled.csv")
    train_data, test_data = prepare_data(data)
    # Q1(train_data)
    train_data, test_data = Q2(train_data, test_data)

    X, Y = prepare_x_train_y_train(train_data)
    Xtest, Ytest = prepare_x_train_y_train(test_data)
    best_mse = 10000
    best_i = None
    best_j = None

    poly = PolynomialFeatures(degree=2, include_bias=False)
    train_data = poly.fit_transform(X)
    polynominal_data = pd.DataFrame(train_data, columns=poly.get_feature_names(X.columns))

    tmp = poly.fit_transform(Xtest)
    polynominal_test_data = pd.DataFrame(tmp, columns=poly.get_feature_names(Xtest.columns))
    best_mse_poly = 999999
    best_i_poly = None
    best_j_poly = None
    for i in range(1,70,7):
        for j in range(2,3):
            regr=CatBoostRegressor()
            poly_reg = CatBoostRegressor()
            # regr = RandomForestRegressor(n_estimators=139,max_depth=i)
            # poly_reg = RandomForestRegressor(n_estimators=139,max_depth=i)
            poly_reg.fit(polynominal_data,Y)
            regr.fit(X, Y)
            res = regr.predict(Xtest)
            poly_res = poly_reg.predict(polynominal_test_data)
            if mean_squared_error(Ytest, res) < best_mse:
                best_mse = mean_squared_error(Ytest, res)
                best_i = i
                best_j = j
            if mean_squared_error(Ytest, poly_res) < best_mse_poly:
                best_mse_poly = mean_squared_error(Ytest, poly_res)
                best_i_poly = i
                best_j_poly = j
            print(f"mse for {i} and {j} regressor : ", mean_squared_error(Ytest, res))
            print(f"mse for {i} and {j} POLY regressor : ", mean_squared_error(Ytest, poly_res))

    print("best is mse " +str(best_mse)+" best i,j " +str(best_i)+"  "+str(best_j))
    print("POLYPOLY best is mse " + str(best_mse_poly) + " best i,j " + str(best_i_poly) + "  " + str(best_j_poly))

    best_layer = None
    best_model = None
    best_mse = 100000
    # for model in {'relu','logistic','identity','tanh'}:
    #     for layers in {(9,11,7,3),(9),(100,25,7,3),(8,4),(100,75,24,11,3),(150,91,33,23,13)
    #         ,(300,150,33,23,13), (21,14),(21,14,7),(21,14,7),(21,19,17,13,11,7,5,3)}:
    #
    #         neural_regressor = MLPRegressor(hidden_layer_sizes=layers,activation=model,learning_rate='constant',learning_rate_init=0.0001)
    #         neural_regressor.fit(X,Y)
    #         if mean_squared_error(Ytest, neural_regressor.predict(Xtest)) < best_mse:
    #             best_mse = mean_squared_error(Ytest, neural_regressor.predict(Xtest))
    #             best_layer = layers
    #             best_model = model
    #         print(f"for {model} and layers {layers} the score is {mean_squared_error(Ytest, neural_regressor.predict(Xtest))}")
    #

    # for model in {'relu','logistic','identity','tanh'}:
    #     print("next model")
    #     for i in range(1,1000,50):
    #         print("next batch")
    #         for j in range(1,1000,50):
    #             layers = (j,i)
    #             neural_regressor = MLPRegressor(hidden_layer_sizes=layers,activation=model,learning_rate='constant',learning_rate_init=0.0001)
    #             neural_regressor.fit(X,Y)
    #             if mean_squared_error(Ytest, neural_regressor.predict(Xtest)) < best_mse:
    #                 best_mse = mean_squared_error(Ytest, neural_regressor.predict(Xtest))
    #                 best_layer = layers
    #                 best_model = model
    #             # print(f"for {model} and layers {layers} the score is {mean_squared_error(Ytest, neural_regressor.predict(Xtest))}")


    best_mse_poly = 100000
    best_layer_poly = None
    best_model_poly = None
    # for model in {'relu', 'logistic', 'identity', 'tanh'}:
    #     print("next model")
    #     for i in range(750, 800, 50):
    #         print("next batch")
    #         for j in range(1, i+1, 65):
    #             for k in range(1, j + 1, 65):
    #
    #                 layers = (i,j,k)
    #                 neural_regressor = MLPRegressor(hidden_layer_sizes=layers, activation=model, learning_rate='constant',
    #                                                 learning_rate_init=0.0001)
    #                 neural_regressor.fit(polynominal_data, Y)
    #                 if mean_squared_error(Ytest, neural_regressor.predict(polynominal_test_data)) < best_mse:
    #                     best_mse_poly = mean_squared_error(Ytest, neural_regressor.predict(polynominal_test_data))
    #                     best_layer_poly = layers
    #                     best_model_poly = model
                # print(
                #     f"POLY model = {model} ,layers= {layers} ,mse= {mean_squared_error(Ytest, neural_regressor.predict(polynominal_test_data))}")

    # i have 1 output and 21 input
    # less than 42 neurons
    # print(f"normal features = best is {best_model} {best_mse} {best_layer}")
    # print(f"poly features = best is {best_model_poly} {best_mse_poly} {best_layer_poly}")

    # models = LinearDecisionTree(800, 1)
    # models.fit(X, Y)
    # # plot_2d_predictions_borders_for_trained_clf(models, X, Y, "T")
    #
    # models = LinearDecisionTree(0, 2)
    # models.fit(X, Y)
    # plot_2d_predictions_borders_for_trained_clf(models, X, Y)
