

# IMPORTS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prepare_HW3 import prepare_data
import numpy as np
from models import compare_gradients ,test_lr
from sklearn.model_selection import train_test_split , cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
itai_id = 3
or_id = 0




def Q1(train_data):
    x_train = train_data.drop(columns=['VirusScore'])
    y_train = train_data['VirusScore']
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    for i, cur_type in enumerate(['-', 'A', 'B']):
         filter_mask = x_train['blood_type'].str.contains(cur_type)
         sns.histplot(data=y_train[~filter_mask], ax=ax[i], stat="density", kde=True,
         line_kws={"linewidth": 3}, color="orange", label="not '{}'".format(cur_type))
         sns.histplot(data=y_train[filter_mask], ax=ax[i], stat="density", kde=True,
         line_kws={"linewidth": 3}, label=cur_type)
         ax[i].set_title("Blood type contains " + cur_type)
         ax[i].legend(), ax[i].grid(alpha=0.5)

    plt.show()


def Q2(train_data,test_data = None):
    # REMOVE BloodType AND ADD NEW feature
    train_data['blood_viruse'] = np.where((train_data['blood_type_AB-']==1)|(train_data['blood_type_A-']==1) | (train_data['blood_type_A+']==1)  ,1 ,0)
    train_data = train_data.drop(['blood_type',
                'blood_type_AB-',
                'blood_type_B+',
                'blood_type_B-',
                'blood_type_O+',
                'blood_type_O-',
                'blood_type_A+',
                'blood_type_A-', 'blood_type_AB+'],axis=1 , errors='ignore')
    test_data['blood_viruse'] = np.where((test_data['blood_type_AB-']==1)|(test_data['blood_type_A-'] == 1) | (test_data['blood_type_A+'] == 1), 1, 0)
    test_data = test_data.drop(['blood_type',
                'blood_type_AB-',
                'blood_type_B+',
                'blood_type_B-',
                'blood_type_O+',
                'blood_type_O-',
                'blood_type_A+',
                'blood_type_A-', 'blood_type_AB+'],axis=1, errors='ignore')
    return train_data,test_data



def Q4(train_data):
    train_subset, train_subset_test = train_test_split(train_data, train_size=0.8, random_state=itai_id + or_id)
    X_train = train_subset.copy()
    X_train.pop('VirusScore')
    y_train = train_subset['VirusScore'].values
    X_train = X_train.values
    compare_gradients(X_train, y_train, deltas=np.logspace(-7, -2, 9))


def Q5(train):
    train_subset , train_subset_test = train_test_split(train,train_size=0.8 , random_state= itai_id+or_id)
    X_train = train_subset.copy()
    X_train.pop('VirusScore')
    y_train = train_subset['VirusScore'].values
    X_train = X_train
    X_val = train_subset_test.copy()
    X_val.pop('VirusScore')
    X_val = X_val.values
    y_val = train_subset_test['VirusScore'].values
    test_lr(X_train.values, y_train, X_val, y_val)
    plt.show()

def prepare_x_train_y_train(train):
    # train_subset, train_subset_test = train_test_split(train, train_size=0.8, random_state=itai_id + or_id)

    X_train = train.copy()
    y_train = None
    if 'VirusScore' in X_train.columns:
        X_train.pop('VirusScore')
        y_train = train['VirusScore'].values
    X_train = X_train
    return X_train, y_train


def Q6(train):
    X_train, y_train = prepare_x_train_y_train(train)

    dummy = DummyRegressor()
    dummy.fit(X_train,y_train)
    scores = cross_validate(dummy,X_train,y_train, scoring="neg_mean_squared_error",cv =5 ,return_train_score=True)
    train_score = np.abs(np.mean(scores['train_score']))
    test_score = np.abs(np.mean(scores['test_score']))
    print(train_score , test_score)
    return train_score , test_score


def get_train_valid_mse(model , X_train, y_train , split = 5):
    # model.fit(X_train,y_train)
    scores = cross_validate(model, X_train, y_train, scoring="neg_mean_squared_error", cv=5, return_train_score=True)
    train_score = np.abs(np.mean(scores['train_score']))
    test_score = np.abs(np.mean(scores['test_score']))
    return train_score, test_score

def dummyRegressor(X_train, y_train):
    # X_train, y_train = prepare_x_train_y_train(train)

    return get_train_valid_mse(DummyRegressor(),X_train,y_train)


def get_best_alpha_and_graph(train, regressor, y_train = None, alpha_sampling =np.logspace(-5, 5, 100)
                             ,title = "Ridge Liner regression optimal strength",doprint = False):
    if y_train is None:
        X_train, y_train = prepare_x_train_y_train(train)
    else:
        X_train = train
    dummy_train, dummy_val = dummyRegressor(X_train,y_train)
    # scores = cross_validate(dummy, X_train, y_train, scoring="neg_mean_squared_error", cv=5, return_train_score=True)
    # train_score = np.mean(scores['train_score'])
    # test_score = np.mean(scores['test_score'])
    eval_train, eval_val = [], []
    for alpha in alpha_sampling:
        etraining, evalid = get_train_valid_mse(regressor(alpha=alpha), X_train, y_train)
        eval_train.append(etraining)
        eval_val.append(evalid)
    best_alpha = alpha_sampling[np.argmin(np.array(eval_val))]
    # best_alpha = alpha_sampling[np.argmin(np.array(eval_train))]
    if doprint:
        plt.semilogx(alpha_sampling, eval_train)
        plt.semilogx(alpha_sampling, eval_val)
        plt.semilogx(alpha_sampling, [dummy_val for _ in alpha_sampling])
        plt.legend(["eval train", "eval validation", "dummy"])
        plt.xlabel("alpha")
        plt.ylabel("error")
        plt.title(title)
        plt.grid()
        plt.show()
        print("best for train: error= ", np.min(np.array(eval_train)), "alpha= ",
              alpha_sampling[np.argmin(np.array(eval_train))])
        print("best for validation: error= ", np.min(np.array(eval_val)), "alpha= ",
              alpha_sampling[np.argmin(np.array(eval_val))])
    return best_alpha

def Q7(train,Title= "Ridge Liner regression optimal strength"):
    return get_best_alpha_and_graph(train,Ridge,doprint=True,title=Title)

# best ridge
def Q8(train):
    X_train, y_train = prepare_x_train_y_train(train)
    best_alpha = get_best_alpha_and_graph(train,Ridge)
    model = Ridge(alpha=best_alpha)
    print(get_train_valid_mse(model, X_train, y_train))
    # return get_train_valid_mse(model, X_train, y_train)

def get_top_5_features_coef(train,model):
    X_train, y_train = prepare_x_train_y_train(train)
    best_train_alpha = get_best_alpha_and_graph(train,model)
    new_reg = model(best_train_alpha)
    new_reg.fit(X_train, y_train)

    tmp_arr = np.abs(np.array(new_reg.coef_))
    indexes = tmp_arr.argsort()[::-1]
    features = indexes[:5]
    return features


def Q9(train):
    X_train, y_train = prepare_x_train_y_train(train)
    indexes =get_top_5_features_coef(train,Ridge)
    for i in indexes[:5]:
        print(X_train.columns[i])
    pass


def absolute_coef_plot(train, model):
    X_train, y_train = prepare_x_train_y_train(train)
    best_train_alpha = get_best_alpha_and_graph(train, model, alpha_sampling=np.logspace(-5, 5, 100))
    new_reg = model(best_train_alpha)
    new_reg.fit(X_train, y_train)

    tmp_arr = abs(new_reg.coef_)
    tmp_arr = np.sort(tmp_arr)[::-1]

    plt.xlabel("index")
    plt.ylabel("absolute value")
    plt.title("Feature absolute values -" + str(model.__name__))
    plt.plot(range(len(tmp_arr)), tmp_arr)
    plt.grid()
    plt.show()


def Q10(train):
    absolute_coef_plot(train,Ridge)

def Q11(train):
    best_alpha =  get_best_alpha_and_graph(train, Lasso,title="Lasso Liner regression optimal strength", doprint=True)


#not a code question (graph one)
def Q12(train):
    pass

# not a code question
def Q13(train):
    X_train, y_train = prepare_x_train_y_train(train)
    best_alpha = get_best_alpha_and_graph(train,Lasso)
    model = Lasso(alpha=best_alpha)
    print(get_train_valid_mse(model, X_train, y_train))

# get best 5 features for lasso regressor
def Q14(train):
    X_train, y_train = prepare_x_train_y_train(train)
    indexes =get_top_5_features_coef(train,Lasso)
    for i in indexes[:5]:
        print(X_train.columns[i])
    pass


def Q15(train):
    absolute_coef_plot(train,Lasso)


def Section6(train, test):
    new_train = train.copy()
    new_test = test.copy()
    poly = PolynomialFeatures(degree=2,include_bias=False)
    train_data = poly.fit_transform(new_train)
    train_data = pd.DataFrame(train_data, columns=poly.get_feature_names(new_train.columns))
    Q6(train_data)
    Q7(train_data)
    Q8(train_data)
    Q9(train_data)
    Q10(train_data)
    Q11(train_data)
    Q12(train_data)
    Q13(train_data)
    Q14(train_data)
    Q15(train_data)

def Q18(train):
    new_train = train.copy()
    poly = PolynomialFeatures(degree=2, include_bias=False)

    new_t = new_train.drop(columns=["VirusScore"])
    train_data = poly.fit_transform(new_t)
    train_data = pd.DataFrame(train_data, columns=poly.get_feature_names(new_t.columns))
    # new_train.set_index(train_data.index)
    train_data = train_data.merge(new_train)
    Q7(train_data, Title="Ridge polynomial regression optimal strength")
    Q8(train_data)
    Q9(train_data)
    Q10(train_data)
    return train_data



from sklearn.metrics import mean_squared_error

def Q20_preparation(train_data,test_data):
    X_train = train_data.copy()
    X_train.pop('VirusScore')
    y_train = train_data['VirusScore'].values

    dummy_model = DummyRegressor(strategy='mean')
    dummy_model = dummy_model.fit(X_train,y_train)
    ridge_model = Ridge(get_best_alpha_and_graph(X_train,Ridge,y_train=y_train)).fit(X_train,y_train)
    lasso_model = Lasso(get_best_alpha_and_graph(X_train,Lasso,y_train=y_train)).fit(X_train,y_train)
    #prepare poly data
    poly = PolynomialFeatures(degree=2, include_bias=False)
    train_data = poly.fit_transform(X_train)
    polynominal_data = pd.DataFrame(train_data, columns=poly.get_feature_names(X_train.columns))

    polynomial_ridge_model = Ridge(get_best_alpha_and_graph(polynominal_data,Ridge,y_train=y_train)).fit(polynominal_data,y_train)

    X_test = test_data.copy()
    X_test.pop('VirusScore')
    y_test = test_data['VirusScore'].values
    train_data = poly.fit_transform(X_test)
    polynominal_test_data = pd.DataFrame(train_data, columns=poly.get_feature_names(X_test.columns))

    dummy_res = dummy_model.predict(X_test)
    ridge_res = ridge_model.predict(X_test)
    lasso_res = lasso_model.predict(X_test)
    polynomial_ridge_res = polynomial_ridge_model.predict(polynominal_test_data)
    print("dummy mse " , mean_squared_error(y_test,dummy_res))
    print("ridge mse ", mean_squared_error(y_test, ridge_res))
    print("lasso mse ", mean_squared_error(y_test, lasso_res))
    print("polynomial ridge mse ", mean_squared_error(y_test, polynomial_ridge_res))
    pass


def prepare_csv(model,data_for_models,name ):
    unlabled_data = pd.read_csv("virus_unlabeled.csv")

    unlabled_data['VirusScore'] = 99
    final_res = pd.DataFrame(unlabled_data['patient_id'])
    unlabled_train, unlabled_test = prepare_data(unlabled_data)
    unlabled_train, unlabled_test = Q2(unlabled_train, unlabled_test)
    unlabled_data = unlabled_train.append([unlabled_test])

    reg = model(get_best_alpha_and_graph(data_for_models, model))
    labled_X, labled_y = prepare_x_train_y_train(data_for_models)

    reg.fit(labled_X, labled_y)
    unlabled_X, unlabled_y = prepare_x_train_y_train(unlabled_data)
    res = reg.predict(unlabled_X)
    final_res['VirusScore'] = res
    final_res.to_csv("pred_"+str(name)+".csv",index=False)

def prepare_poly_csv():
    data = pd.read_csv("virus_labeled.csv")
    train_data, train_data2 = prepare_data(data)
    train_data, train_data2 = Q2(train_data, train_data2)

    train_data = train_data.append([train_data2])

    labled_X , labled_y = prepare_x_train_y_train(train_data)
    # prepare poly data
    poly = PolynomialFeatures(degree=2, include_bias=False)
    train_data = poly.fit_transform(labled_X)


    polynominal_data = pd.DataFrame(train_data, columns=poly.get_feature_names(labled_X.columns))

    polynomial_ridge_model = Ridge(get_best_alpha_and_graph(polynominal_data, Ridge, y_train=labled_y)).fit(
        polynominal_data, labled_y)
    # preparation of unlabled.csv
    unlabled_data = pd.read_csv("virus_unlabeled.csv")

    unlabled_data['VirusScore'] = 99
    final_res = pd.DataFrame(unlabled_data['patient_id'])
    unlabled_train, unlabled_test = prepare_data(unlabled_data)
    unlabled_train, unlabled_test = Q2(unlabled_train, unlabled_test)
    unlabled_data = unlabled_train.append([unlabled_test])

    unlabled_X, unlabled_y = prepare_x_train_y_train(unlabled_data)

    train_data = poly.fit_transform(unlabled_X)
    polynominal_test_data = pd.DataFrame(train_data, columns=poly.get_feature_names(unlabled_X.columns))

    res = polynomial_ridge_model.predict(polynominal_test_data)

    final_res['VirusScore'] = res
    final_res.to_csv("pred_" + str(6) + ".csv", index=False)




def test_h_vs_poly(train,test):

    train_a = train[ train['blood_viruse'] == 1]
    train_b = train[train['blood_viruse'] == 0]

    X_a , y_a = prepare_x_train_y_train(train_a)

    X_b, y_b = prepare_x_train_y_train(train_b)

    model_a = Ridge(get_best_alpha_and_graph(X_a,Ridge,y_train=y_a)).fit(X_a,y_a)
    model_b = Ridge(get_best_alpha_and_graph(X_b,Ridge,y_train=y_b)).fit(X_b,y_b)

    X_test , y_test = prepare_x_train_y_train(test)

    model_a_res = model_a.predict(X_test)
    model_b_res = model_b.predict(X_test)

    print("dummy mse ", mean_squared_error(y_test, model_a_res))
    print("dummy mse ", mean_squared_error(y_test, model_b_res))

    X_train ,y_train = prepare_x_train_y_train(train)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    train_data = poly.fit_transform(X_train)
    polynominal_data = pd.DataFrame(train_data, columns=poly.get_feature_names(X_train.columns))

    polynomial_ridge_model = Ridge(get_best_alpha_and_graph(polynominal_data, Ridge, y_train=y_train)).fit(
        polynominal_data, y_train)
    train_data = poly.fit_transform(X_test)
    polynominal_test_data = pd.DataFrame(train_data, columns=poly.get_feature_names(X_test.columns))

    polynomial_ridge_res = polynomial_ridge_model.predict(polynominal_test_data)

    print("polynomial ridge mse ", mean_squared_error(y_test, polynomial_ridge_res))

if __name__ == '__main__':
    data = pd.read_csv("virus_labeled.csv")
    unlabled = pd.read_csv("virus_unlabeled.csv")
    train_data, test_data = prepare_data(data,unlabled)
    # Q1(train_data)
    train_data, test_data = Q2(train_data, test_data)
    # Q4(train_data=train_data)
    # Q5(train_data)
    # Q6(train_data)
    # Q7(train_data)
    # Q8(train_data)
    # Q9(train_data)
    # Q10(train_data)
    # Q11(train_data)
    # Q12(train_data)
    # Q13(train_data)
    # Q14(train_data)
    # Q15(train_data)
    # Section6(train_data,test_data)
    # Q18(train_data)
    # Q20_preparation(train_data,test_data)


    # data_for_models =train_data.append([test_data])
    # prepare_csv(Ridge,data_for_models,4)
    # prepare_csv(Lasso,data_for_models,5)
    # prepare_poly_csv()
    # test_h_vs_poly(train_data,test_data)
    pass
