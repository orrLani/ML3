
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
# from sklearn.ensemble import
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
# ask if we can use that
import math


##
# previous imports
from feature_engine.imputation import RandomSampleImputer
##
# questions_module = __import__('205819220-308060873')


my_id = 3
or_id = 0
i = 1
IS_TRAINING = False
Q_1_Q_3_TRAINING = dict()
MEAN_MEDIAN_TRAINING = dict()
MOST_FR = dict()

def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])


def meanMedian(x):
    return pd.Series(index=['mean','median'],data=[x.mean(),x.median()])


def turn_binary(data):
    data['sex'] = data['sex'].replace(['F','M'],[0,1])
    data['risk'] = data['risk'].replace(['High','Low'],[1,0])
    data['spread'] = data['spread'].replace(['High', 'Low'],[1, 0])
    return data


def NoneFill(data,col):
    val = data[col].mean()
    data[col].fillna(value=val,inplace=True)


def percent25(df):
    return df.quantile(0.25)


def percent75(df):
    return df.quantile(0.75)



# returns the dataframe after cleaning the column (can work with given bounds)
def CleanIQR(data,col, is_training): # need to add temp df (to not ruin data when addressing 1 value in colum )
    global Q_1_Q_3_TRAINING
    tmp_data = data.copy()
    if is_training:
        q1 = percent25(tmp_data[col])
        q3 = percent75(tmp_data[col])
        Q_1_Q_3_TRAINING['q1'] = q1
        Q_1_Q_3_TRAINING['q3'] = q3

    else:
        q1 = Q_1_Q_3_TRAINING['q1']
        q3 = Q_1_Q_3_TRAINING['q3']


    iqr = q3 - q1
    upper_bound = q3 + 1.5*iqr
    lower_bound = q1 - 1.5*iqr


    tmp_data[col] = np.where(
        tmp_data[col] > upper_bound, upper_bound,
        np.where(tmp_data[col]< lower_bound,
                 lower_bound , tmp_data[col]))
    return tmp_data

# cleans data according to 2 columns
def DuoCleanIQR(data,col,feature,is_training):
    features = data[feature].unique()
    for i in features:
        tmp_data = data.copy()
        # tmp_data = tmp_data[tmp_data.col == i]
        tmp_data = tmp_data[tmp_data[feature] == i]
        tmp_data = CleanIQR(tmp_data, col,is_training)
        # pd.concat([data,tmp_data]).drop_duplicates()
        data.update(tmp_data)
    return data

# returns 3 histograms for feature (histogram of feature with one of the target features)
def get_histograms(data,feature):
    COL_NAME = ['risk','covid','spread']
    COLS = 3
    ROWS = int(np.ceil(3 / COLS))
    for i, column in enumerate(COL_NAME, 1):
        ax = plt.subplot(ROWS, COLS, i)
        ax.title.set_text(column)
        sns.histplot(data=data, x=feature, hue=column, kde=True)
        plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()


def split_dataset(data):
    train, test = train_test_split(data, train_size=0.8, random_state=my_id + or_id)
    train.to_csv("train.csv")
    test.to_csv("test.csv")
    return (train , test)


# splits blood type to OHE vector
def split_blood_type(train_data,test_data):
    train, test = imute_blood_type(train=train_data,test=test_data)
    train_temp = train['blood_type']
    train = pd.get_dummies(data=train, columns=['blood_type'])
    train['blood_type'] = train_temp

    test_temp = test['blood_type']
    test = pd.get_dummies(data=test, columns=['blood_type'])
    test['blood_type'] = test_temp
    # train.to_csv("train.csv")
    return train, test

# splits symptoms into different features and returns the updated
# dataframe
def split_symptoms(data, save_csv = False, name_of_csv=''):
    data = data.join(data['symptoms'].str.get_dummies(sep=';'))
    # train.to_csv("train.csv")

    if save_csv:
        data.to_csv(name_of_csv)
    return data
    #should be return ?
    # not a code question


#
# deleted the use of geo to get data
#


# this function generates 3 historgrams for each feature
# with one of the target features (risk, spread,covid)

def get_all_hist(data):
    for att in data.columns.to_list():
        get_histograms(data,att)




def clear_outliers(data, is_training):
    continuous_cols = ['PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_05',
                       'PCR_06', 'PCR_07', 'PCR_08', 'PCR_09', 'PCR_10']

    # global IS_TRAINING
    # global MAX_MIN_TRAINING
    # if is_training:
    #     IS_TRAINING = True
    #     numric_data = data[continuous_cols + ['sugar_levels', 'sport_activity', 'weight', 'age']]
    #     MAX_MIN_TRAINING = numric_data.apply(minMax)
    # else:
    #     IS_TRAINING = False
    # connection clear
    data = DuoCleanIQR(data, 'sugar_levels', 'sport_activity',is_training)
    data = DuoCleanIQR(data, 'weight', 'age',is_training)

    # continuous columns clear
    for col in continuous_cols:
        data = CleanIQR(data, col,is_training)

    return data




# mean / median
# end of tail
# random sample
# constant value (arbitrary value)
# most frequent

def imute_blood_type(train , test : pd.DataFrame):
    imputer = RandomSampleImputer(random_state=my_id + or_id)
    new_train = train.copy()
    new_test = test.copy()
    new_train['blood_type'] = imputer.fit_transform(train[['blood_type']])
    new_test['blood_type'] = imputer.transform(test[['blood_type']])

    return new_train, new_test


def impute_simpleimputer(data,feature, impute_method):
    imputer = SimpleImputer(missing_values=np.nan, strategy=impute_method)
    data[feature] = imputer.fit_transform(data[[feature]])
    return data


def impute_features(data,is_training):
    median_impute = ['happiness_score','age', 'num_of_siblings', 'conversations_per_day', 'sport_activity', 'PCR_05', 'PCR_10']
    mean_impute = ['household_income', 'sugar_levels', 'PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_06', 'PCR_07', 'PCR_08', 'PCR_09']
    random_impute = ['pcr_date']

    global IS_TRAINING
    global MEAN_MEDIAN_TRAINING
    global MOST_FR
    if is_training:
        IS_TRAINING = True
        relevent_data = data[median_impute + mean_impute]
        MEAN_MEDIAN_TRAINING = relevent_data.apply(meanMedian)
        for i in median_impute:
            impute_simpleimputer(data, i, "median")

        for i in mean_impute:
            impute_simpleimputer(data, i, "mean")
    else:
        IS_TRAINING = False
        for col in median_impute:
            imputer = SimpleImputer(missing_values=np.nan, strategy='constant',
                                    fill_value=MEAN_MEDIAN_TRAINING[col]['median'])
            data[col] = imputer.fit_transform(data[[col]])


        for col in mean_impute:
            imputer = SimpleImputer(missing_values=np.nan, strategy='constant',
                                    fill_value=MEAN_MEDIAN_TRAINING[col]['mean'])
            data[col] = imputer.fit_transform(data[[col]])



    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

    imputer = RandomSampleImputer(random_state=my_id +or_id)
    for i in  random_impute:
        data[i] = imputer.fit_transform(data[[i]])
    weight_list = prepare_weight_age_function(data)
    data['weight'] = data.apply( lambda  row: weight_list[int(row['age'])],axis=1)
    imputer = RandomSampleImputer(random_state=my_id + or_id)
    data['sex'] = imputer.fit_transform(data[['sex']])
    return data





def features_to_keep(data):
    remaining_features = [ 'age', 'sex', 'num_of_siblings','conversations_per_day','household_income',
                           'sugar_levels', 'sport_activity',  'PCR_01', 'PCR_02', 'PCR_03',
                          'PCR_04', 'PCR_05', 'PCR_06','PCR_07', 'PCR_08','PCR_09', 'PCR_10', 'blood_type',
                           'blood_type_AB-', 'blood_type_B+',
                           'blood_type_B-','blood_type_O+', 'blood_type_O-','cough','blood_type_A+','blood_type_A-',
                           'fever','shortness_of_breath', 'VirusScore'
                           ]

    data = data[remaining_features]

    return data

def prepare_weight_age_function(data):
    ret_list = []
    for i in range (0,100):
        tmp_data = data[ data['age'] == i ]
        ret_list.append(tmp_data['weight'].mean())
    return ret_list


def remove_unwanted_features(data):
    toRemove = set()
    for i in data:
        if 'NAMED' in str(i).upper():
            toRemove.add(i)
    for i in toRemove:
        data.drop(i, axis=1, inplace=True)
    return data


#######################################
def normalize(trainset,testset,row,process):
    temp = trainset[row].values.reshape(-1, 1)
    trainset[row] = process.fit_transform(temp)
    testset[row] = process.transform(testset[row].values.reshape(-1,1))
    return trainset,testset

def convertToNumber(s):
    return int.from_bytes(str(s).encode(), 'little')

def convertToBinary(bool_condition):
        if bool_condition:
            return 1
        return -1
def create_number_convention(df):
    df['sex'] = df['sex'].map(dict(F=-1,M=1))
    # targets
    # covid
    # df['covid'] = df['covid'].apply(convertToBinary)
    # # risk
    # df['risk'] = df['risk'].map(dict(Low=-1, High=1))
    # # spread
    # df['spread'] = df['spread'].map(dict(Low=-1, High=1))

    # drop date
    return df


def normalize_data(train:pd.DataFrame,test : pd.DataFrame):
    train,test = normalize(trainset=train,testset=test, row='age', process=preprocessing.MinMaxScaler())
    train,test =normalize(trainset=train,testset=test, row='num_of_siblings', process=preprocessing.StandardScaler())
    train,test =normalize(trainset=train,testset=test, row='conversations_per_day', process=preprocessing.MinMaxScaler())
    train,test =normalize(trainset=train,testset=test, row='household_income', process=preprocessing.MinMaxScaler())
    train,test =normalize(trainset=train,testset=test, row='sugar_levels', process=preprocessing.StandardScaler())
    train,test =normalize(trainset=train,testset=test, row='sport_activity', process=preprocessing.MinMaxScaler())

    train,test = normalize(trainset=train,testset=test, row='PCR_01', process=preprocessing.MinMaxScaler())
    train,test = normalize(trainset=train,testset=test, row='PCR_02', process=preprocessing.MinMaxScaler())
    train,test = normalize(trainset=train,testset=test, row='PCR_03', process=preprocessing.StandardScaler())
    train,test = normalize(trainset=train,testset=test, row='PCR_04', process=preprocessing.MinMaxScaler())
    train, test = normalize(trainset=train, testset=test, row='PCR_05', process=preprocessing.MinMaxScaler())
    train,test = normalize(trainset=train,testset=test, row='PCR_06', process=preprocessing.MinMaxScaler())
    train,test = normalize(trainset=train,testset=test, row='PCR_07', process=preprocessing.MinMaxScaler())
    train,test = normalize(trainset=train,testset=test, row='PCR_08',process=preprocessing.MinMaxScaler())
    train,test = normalize(trainset=train,testset=test, row='PCR_09',process=preprocessing.MinMaxScaler())
    train,test = normalize(trainset=train,testset=test, row= 'PCR_10',process=preprocessing.MinMaxScaler())

    return train,test



def _prepare_data(train_data: pd.DataFrame,test_data:pd.DataFrame):
    """
    :param data:
    :param training_data:
    :return:
    """
    # data = data.copy()
    # Q6 change the blood_type facture to OHE
    # data = questions_module.Q6(data=data)
    train_data, test_data = split_blood_type(train_data=train_data,test_data=test_data)

    # Q7 change the symptoms facture to OHE
    # data = questions_module.Q7(data=data,save_csv=False)
    train_data = split_symptoms(data=train_data)
    test_data = split_symptoms(data=test_data)


    train_data = clear_outliers(data=train_data, is_training=True)
    test_data = clear_outliers(data=test_data, is_training=False)

    # TODO FIX BY THE TRAINING DATA
    # data = questions_module.Q11(data=data, plot=False, is_training=False)
    # Q17 missing values

    train_data = impute_features(data=train_data,is_training=True)
    test_data = impute_features(data=test_data, is_training=False)

    # Q23 clean feactures
    train_data = features_to_keep(data=train_data)
    test_data = features_to_keep(data=test_data)
    # data = questions_module.Q23(data=data,save_csv=False,name_csv ='data_clean')

    # data = normalize_data(data=data)
    # normalize the data

    return train_data,test_data





def imute_blood_type(train : pd.DataFrame, test : pd.DataFrame):
    imputer = RandomSampleImputer(random_state=my_id + or_id)
    new_train = train.copy()
    new_test = test.copy()
    new_train['blood_type'] = imputer.fit_transform(train[['blood_type']])
    new_test['blood_type'] = imputer.transform(test[['blood_type']])
    return new_train, new_test


def prepare_data( data : pd.DataFrame ):
    data = create_number_convention(data)
    # data = pd.read_csv("training.csv")

    train_data, test_data = split_dataset(data)
    train,test = _prepare_data(train_data=train_data,test_data=test_data)
    train,test = normalize_data(train,test)
    # train = remove_unwanted_features(train)
    # test = remove_unwanted_features(test)
    pd.DataFrame(train).to_csv("train_prepared.csv",index=False)
    pd.DataFrame(test).to_csv("test_prepared.csv",index=False)

    return train , test


import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    data = pd.read_csv("virus_labeled.csv",index_col = False)
    train, test = prepare_data(data=data)
    pass
    # train = Q1(train_data=train)
    # Q1(train_data=train)
