

# IMPORTS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prepare_HW3 import prepare_data
import numpy as np
from models import compare_gradients ,test_lr
from sklearn.model_selection import train_test_split , cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
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




def Q2(train_data,test_data):
    # REMOVE BloodType AND ADD NEW feature
    train_data['blood_viruse'] = np.where((train_data['blood_type_A-']==1) | (train_data['blood_type_A+']==1)  ,1 ,0)
    train_data = train_data.drop(['blood_type',
                'blood_type_AB-',
                'blood_type_B+',
                'blood_type_B-',
                'blood_type_O+',
                'blood_type_O-',
                'blood_type_A+',
                'blood_type_A-'],axis=1)
    test_data = test_data.drop(['blood_type',
                'blood_type_AB-',
                'blood_type_B+',
                'blood_type_B-',
                'blood_type_O+',
                'blood_type_O-',
                'blood_type_A+',
                'blood_type_A-'],axis=1)
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

def prepare_x_train_y_train(train):
    train_subset, train_subset_test = train_test_split(train, train_size=0.8, random_state=itai_id + or_id)
    X_train = train_subset.copy()
    X_train.pop('VirusScore')
    y_train = train_subset['VirusScore'].values
    X_train = X_train
    return X_train, y_train


def Q6(train):
    X_train, y_train = prepare_x_train_y_train(train)

    dummy = DummyRegressor()
    dummy.fit(X_train,y_train)
    scores = cross_validate(dummy,X_train,y_train, scoring="neg_mean_squared_error",cv =5 ,return_train_score=True)
    train_score = np.mean(scores['train_score'])
    test_score = np.mean(scores['test_score'])
    return train_score , test_score


def get_train_valid_mse(model , X_train, y_train , split = 5):
    scores = cross_validate(model, X_train, y_train, scoring="neg_mean_squared_error", cv=5, return_train_score=True)
    train_score = np.mean(scores['train_score'])
    test_score = np.mean(scores['test_score'])
    return train_score, test_score

def dummyRegressor(train):
    X_train, y_train = prepare_x_train_y_train(train)

    return get_train_valid_mse(DummyRegressor(),X_train,y_train)


def Q7(train):

    X_train, y_train = prepare_x_train_y_train(train)

    dummy_train , dummy_val = dummyRegressor(train)
    # scores = cross_validate(dummy, X_train, y_train, scoring="neg_mean_squared_error", cv=5, return_train_score=True)
    # train_score = np.mean(scores['train_score'])
    # test_score = np.mean(scores['test_score'])
    alpha_sampling = np.logspace(-5,5,150)
    eval_train, eval_val = [], []
    for alpha in np.logspace(-5,5,150):
        etraining, evalid = get_train_valid_mse(Ridge(alpha=alpha), X_train, y_train, 5)
        eval_train.append(etraining)
        eval_val.append(evalid)
    plt.semilogx(np.logspace(-5,5,150), eval_train)
    plt.semilogx(np.logspace(-5,5,150), eval_val)
    plt.semilogx(np.logspace(-5,5,150), [dummy_val for _ in np.logspace(-5,5,150)])
    plt.legend(["eval train", "eval val", "dummy"])
    plt.xlabel("alpha")
    plt.ylabel("error")
    # plt.yscale('log')
    plt.grid()
    plt.show()
    best_train_alpha = alpha_sampling[np.argmin(np.array(eval_train))]
    print("best for train: error-", np.min(np.array(eval_train)), "alpha-", alpha_sampling[np.argmin(np.array(eval_train))])
    print("best for validation: error-", np.min(np.array(eval_val)), "alpha-",
          alpha_sampling[np.argmin(np.array(eval_val))])
    return best_train_alpha


def Q9(train):
    X_train, y_train = prepare_x_train_y_train(train)
    best_train_alpha = Q7(train)
    new_reg = Ridge(best_train_alpha)
    new_reg.fit(X_train,y_train)

    tmp_arr = np.abs(np.array(new_reg.coef_))
    indexes = tmp_arr.argsort()[::-1]
    for i in indexes[:5]:
        print(X_train.columns[i])
    pass

def Q10(train):


if __name__ == '__main__':
    data = pd.read_csv("virus_labeled.csv")
    train_data, test_data = prepare_data(data)
    train_data, test_data = Q2(train_data, test_data)
    # Q4(train_data=train_data)
    # Q5(train_data)
    # Q6(train_data)
    # Q7(train_data)
    Q9(train_data)
