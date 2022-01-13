

# IMPORTS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prepare_HW3 import prepare_data
import numpy as np
from models import compare_gradients ,test_lr
from sklearn.model_selection import train_test_split
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


if __name__ == '__main__':
    data = pd.read_csv("virus_labeled.csv")
    train_data, test_data = prepare_data(data)
    train_data, test_data = Q2(train_data, test_data)
    Q4(train_data=train_data)
    Q5(train_data)



