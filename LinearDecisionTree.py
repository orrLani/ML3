from sklearn.svm import LinearSVC, SVC
import numpy as np
# from paintings import *
# from auxiliary_functions import plot_2d_predictions_borders_for_trained_clf
from collections import Counter

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
    # for i in range(50,150):
    #     for j in range(2,5):
    #         regr = RandomForestRegressor(n_estimators=i,min_samples_split=j)
    #         regr.fit(X, Y)
    #         res = regr.predict(Xtest)
    #         if mean_squared_error(Ytest, res) < best_mse:
    #             best_mse = mean_squared_error(Ytest, res)
    #             best_i = i
    #             best_j = j
    #         # print("mse for i and j regressor : ", mean_squared_error(Ytest, res))
    # print("best is mse " +str(best_mse)+" best i,j " +str(best_i)+"  "+str(best_j))


    for model in {'relu','logistic','identity','tanh'}:
        for layers in {(9,11,7,3),(9),(100,25,7,3),(8,4),(100,75,24,11,3),(150,91,33,23,13),(300,150,33,23,13)}:

            neural_regressor = MLPRegressor(hidden_layer_sizes=layers,activation=model,learning_rate='constant',learning_rate_init=0.0001)
            neural_regressor.fit(X,Y)
            print(f"for {model} and layers {layers} the score is {mean_squared_error(Ytest, neural_regressor.predict(Xtest))}")








    # models = LinearDecisionTree(800, 1)
    # models.fit(X, Y)
    # # plot_2d_predictions_borders_for_trained_clf(models, X, Y, "T")
    #
    # models = LinearDecisionTree(0, 2)
    # models.fit(X, Y)
    # plot_2d_predictions_borders_for_trained_clf(models, X, Y)
