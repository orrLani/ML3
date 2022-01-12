
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

# questions_module = __import__('205819220-308060873')



def normalize(df,row,process):
    temp = df[row].values.reshape(-1, 1)
    df[row] = process.fit_transform(temp)

    return df

def convertToNumber(s):
    return int.from_bytes(str(s).encode(), 'little')

def convertToBinary(bool_condition):
        if bool_condition:
            return 1
        return -1
def create_number_convention(df):
    # sex, pcr_dat, country, postcode , is_army , home_country , risk
    # sex
    df['sex'] = df['sex'].map(dict(F=-1,M=1))

    # country
    df['country'] = df['country'].apply(convertToNumber)

    # postcode
    df['postcode'] = df['postcode'].apply(convertToNumber)

    # is army
    df['is_army'] = df['is_army'].apply(convertToBinary)

    # home_country
    df['home_country'] = df['home_country'].apply(convertToNumber)

    # targets
    # covid
    df['covid'] = df['covid'].apply(convertToBinary)
    # risk
    df['risk'] = df['risk'].map(dict(Low=-1, High=1))
    # spread
    df['spread'] = df['spread'].map(dict(Low=-1, High=1))

    # drop date
    df = df.drop(['pcr_date','postcode','home_country','country'], axis=1)




    return df



def normalize_data(data:pd.DataFrame):
    data = normalize(df=data, row='age', process=preprocessing.MinMaxScaler())
    data =normalize(df=data, row='num_of_siblings', process=preprocessing.StandardScaler())
    data =normalize(df=data, row='conversations_per_day', process=preprocessing.MinMaxScaler())
    data =normalize(df=data, row='household_income', process=preprocessing.MinMaxScaler())
    # data = normalize(df=data, row='household_income', process=preprocessing.StandardScaler())
    data =normalize(df=data, row='sugar_levels', process=preprocessing.StandardScaler())
    data =normalize(df=data, row='sport_activity', process=preprocessing.MinMaxScaler())

    data = normalize(df=data, row='PCR_01', process=preprocessing.StandardScaler())
    data = normalize(df=data, row='PCR_02', process=preprocessing.StandardScaler())
    data = normalize(df=data, row='PCR_03', process=preprocessing.StandardScaler())
    data = normalize(df=data, row='PCR_04', process=preprocessing.MinMaxScaler())
    data = normalize(df=data, row='PCR_06', process=preprocessing.StandardScaler())
    data = normalize(df=data, row='PCR_07', process=preprocessing.StandardScaler())
    data = normalize(df=data,row='PCR_08',process=preprocessing.MinMaxScaler())
    data = normalize(df=data,row='PCR_09',process=preprocessing.MinMaxScaler())
    data = normalize(df=data,row= 'PCR_10',process=preprocessing.StandardScaler())

    return data



def preprare_data(data:pd.DataFrame, training_data: pd.DataFrame):
    """
    :param data:
    :param training_data:
    :return:
    """
    # data = data.copy()

    # Q6 change the blood_type facture to OHE
    # data = questions_module.Q6(data=data)
    training_data = questions_module.Q6(data=training_data)

    # Q7 change the symptoms facture to OHE
    # data = questions_module.Q7(data=data,save_csv=False)
    training_data = questions_module.Q7(data=training_data)

    # Q8 craft new features and add them to the dataset
    ##training_data = questions_module.Q8(data=training_data,save_csv=True,name_csv='training')
    # data = questions_module.Q8(data=data,save_csv=True,name_csv='data')


    # Q8 mereg data and create ohe
    training_new_features_to_add = pd.read_csv('training_new_features_to_add.csv')
    training_data = questions_module.Q8_merge_data(data_first=training_data,
                                                                  data_second=training_new_features_to_add,
                                                                  save_csv=False,
                                                                  name_csv="training")
    #
    # data_new_features_to_add = pd.read_csv('data_new_features_to_add.csv')
    # data = questions_module.Q8_merge_data(data_first=data,
    #                                                           data_second= data_new_features_to_add,
    #                                                           save_csv=True,
    #                                                           name_csv='data')
    #
    # Q11 clean IQR
    training_data = questions_module.Q11(data=training_data,plot=False,is_training=True)
    # TODO FIX BY THE TRAINING DATA
    # data = questions_module.Q11(data=data, plot=False, is_training=False)
    data = questions_module.Q11(data=data, plot=False, is_training=False)

    # Q17 missing values
    training_data = questions_module.Q17(data=training_data,plot=False,is_training=True,save_csv=False,
    name_csv ='training_fill')
    # data = questions_module.Q17(data=data, plot=False, is_training=False,save_csv=False)


    # Q23 clean feactures
    training_data = questions_module.Q23(data=training_data,save_csv=True,name_csv ='train_clean.csv')
    # data = questions_module.Q23(data=data,save_csv=False,name_csv ='data_clean')

    training_data = normalize_data(data=training_data)
    # normalize the data

    return training_data



def prepare_data(data,is_trainging):
    # splits blood type to OHE vector
    data = pd.get_dummies(data=data, columns=['blood_type'])

    # splits symptoms into different features and returns the updated
    data = data.join(data['symptoms'].str.get_dummies(sep=';'))

    # returns the dataframe after cleaning the features from outliers
    continuous_cols = ['PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_05',
                       'PCR_06', 'PCR_07', 'PCR_08', 'PCR_09', 'PCR_10']

    global IS_TRAINING
    global MAX_MIN_TRAINING
    if is_training:
        IS_TRAINING = True
        numric_data = data[continuous_cols + ['sugar_levels', 'sport_activity', 'weight', 'age']]
        MAX_MIN_TRAINING = numric_data.apply(minMax)
    else:
        IS_TRAINING = False
    if plot:
        plt.subplot(2, 2, 1)
        sns.boxplot(data=data, x='risk', y='household_income')
        plt.show()

    # connection clear
    data = DuoCleanIQR(data, 'sugar_levels', 'sport_activity')
    data = DuoCleanIQR(data, 'weight', 'age')

    # continuous columns clear
    for col in continuous_cols:
        data = CleanIQR(data, col)

    if plot:
        Q11_plots(data)

    return data










if __name__ == '__main__':
    data = pd.read_csv("virus_data.csv")
    # data = pd.read_csv("training.csv")
    train_data, test_data = questions_module.Q5(data)
    train = preprare_data(training_data=train_data,data=train_data)
    train['covid'] = (train['covid'].astype(int)*2-1)
    pd.DataFrame(train).to_csv("trained.csv")

    print(train['covid'])
