import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
# from sklearn.ensemble import
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from feature_engine.imputation import RandomSampleImputer
from geopy.geocoders import Nominatim
my_id = 3
or_id = 0
i = 1
IS_TRAINING = False
MAX_MIN_TRAINING = None
MEAN_MEDIAN_TRAINING = None
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


def percent25(df):
    return df.quantile(0.25)


def percent75(df):
    return df.quantile(0.75)




#
# returns the dataframe after cleaning the column (can work with given bounds)
def CleanIQR(data,col, upper_bound = None ,lower_bound=None): # need to add temp df (to not ruin data when addressing 1 value in colum )
    tmp_data = data.copy()
    q1 = percent25(tmp_data[col])
    q3 = percent75(tmp_data[col])
    iqr = q3 - q1
    if upper_bound is None:
        upper_bound = q3 + 1.5*iqr
    if lower_bound is None:
        lower_bound = q1 - 1.5*iqr

    global IS_TRAINING
    global MAX_MIN_TRAINING

    if not IS_TRAINING:
        upper_bound = MAX_MIN_TRAINING[col]['max']
        lower_bound = MAX_MIN_TRAINING[col]['min']

    tmp_data[col] = np.where(
        tmp_data[col] > upper_bound, upper_bound,
        np.where(tmp_data[col]< lower_bound,
                 lower_bound , tmp_data[col]))
    return tmp_data

# cleans data according to 2 columns
def DuoCleanIQR(data,col,feature,):
    features = data[feature].unique()
    for i in features:
        tmp_data = data.copy()
        # tmp_data = tmp_data[tmp_data.col == i]
        tmp_data = tmp_data[tmp_data[feature] == i]
        tmp_data = CleanIQR(tmp_data, col)
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


def Q1(data):
    print(str(data.shape[0]) + " is the number of rows")
    print(str(data.shape[1]) + " is the number of columns")

def Q2(data):
    print(data['num_of_siblings'].value_counts())

def Q3(data):
    pass
    # not a code question

def Q4(data):
    pass
    # not a code question


# splits to train and test set according to our id sum
def Q5(data):
    train,test = train_test_split(data, train_size=0.8, random_state=my_id + or_id)
    train.to_csv("train.csv")
    test.to_csv("test.csv")
    return (train , test)

# splits blood type to OHE vector
def Q6(data):
    data =pd.get_dummies(data=data, columns=['blood_type'])
    # train.to_csv("train.csv")
    return data
    # print(x)

# splits symptoms into different features and returns the updated
# dataframe
def Q7(data,save_csv = False,name_of_csv=''):
    data = data.join(data['symptoms'].str.get_dummies(sep=';'))
    # train.to_csv("train.csv")

    if save_csv:
        data.to_csv(name_of_csv)
    return data
    #should be return ?
    # not a code question


# recieves coordiantes and extracts from it
# the x and y and returns the reverse action of i
# ( returns alot of information based on the coordiantes)
# such as country, postcode,  street etc
from geopy.geocoders import Nominatim
def Q8_transform_coordinates(coordinates):

    if coordinates is np.nan:
        return None
    geolocator = Nominatim(user_agent= "myGecoder")
    coordinates = coordinates.replace("(","")
    coordinates = coordinates.replace(")", "")
    coordinates = coordinates.replace("\'", "")
    (x ,y ) = coordinates.split(",")
    location = geolocator.reverse((x,y), language="en")

    return location

# this function generates 3 historgrams for each feature
# with one of the target features (risk, spread,covid)

def get_all_hist(data):
    for att in data.columns.to_list():
        get_histograms(data,att)

# this funcitons merges back the data we made into one
def Q8_merge_data(data_first,data_second,save_csv=False,name_csv=''):
    result_data:pd.DataFrame = pd.concat([data_first,data_second],axis=1,join='inner')
    if save_csv:
        result_data.to_csv(f'{name_csv}_after_Q8_and_cleaning_UNMATTED_cols.csv')
    return result_data

# this functions extracts features from address and current location
# the extracted features are:
# is_army : if the entry is serving in the army (according to address)
# home country : returns the home country of the entry
# country : returns current country the entry is at (can be used with home country to check if
# its a toursit)
# postcode: postcode of current location , which can be used to get alot of information if needed
# return the dataframe with the newly extracted features
def Q8(data, save_csv=False, name_csv=''):
    new_data = pd.DataFrame()
    is_army = []
    country_residence = []
    process = 0
    for i in data.index:
        print(f' finish {100 * process / len(data.index)}%')
        process += 1
        if data['address'][i] is not np.nan:
            val = data['address'][i]
            if "Unit" in str(val) or "USNS" in str(val) or "USNV" in str(val):
                is_army.append(True)
            else:
                is_army.append(False)
            if "," in str(val):
                tmp = str(val).split(',')[1]
                tmp = tmp.strip().split(' ')[0]
                country_residence.append(tmp)
            else:
                country_residence.append(np.nan)
        else:
            is_army.append(np.nan)
            country_residence.append(np.nan)

        if data['current_location'][i] is not np.nan:
            info_dict = Q8_transform_coordinates(data['current_location'][i])
            info_dict = info_dict.raw['address']
            relvent_dict = dict()
            if 'country' in info_dict.keys():
                relvent_dict['country'] = info_dict['country']
            else:
                relvent_dict['country'] = np.nan
            if 'county' in info_dict.keys():
                relvent_dict['county'] = info_dict['county']
            else:
                relvent_dict['county'] = np.nan
            if 'postcode' in info_dict.keys():
                relvent_dict['postcode'] = info_dict['postcode']
            else:
                relvent_dict['postcode'] = np.nan
            new_data = new_data.append(relvent_dict, ignore_index=True)
        else:
                new_data = new_data.append(pd.Series(), ignore_index=True)
    new_data['is_army'] = is_army
    new_data['home_country'] = country_residence
    data = turn_binary(data)
    # save data to csv
    if save_csv:
        new_data.to_csv(f'{name_csv}.csv')

    merge_data = Q8_merge_data(data_first=data, data_second=new_data)

    return merge_data



# generates histograms for sugar levels
def Q9(data):
    get_histograms(data,"sugar_levels")

# generates the asked histograms for Q11
def Q11_plots(data):
    # SHOW PLOTS
    plt.subplot(1, 3, 2)
    sns.histplot(data=data, x='household_income', hue='risk', kde=True)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

    # plt.subplot(2, 2, 2)
    # sns.boxplot(c['num_of_siblings'])
    # plt.show()
    # plt.subplot(2, 2, 3)
    # sns.boxplot(data['num_of_siblings'])
    # plt.show()

    # unique_values = data['risk'].unique()
    # plt.subplot(2, 2, 1)
    # sns.boxplot(data=data,x='risk',y='household_income')
    # plt.show()
    plt.subplot(1, 1, 1)
    sns.boxplot(data=data, x='age', y='weight', width= .5)
    plt.show()
    plt.subplot(2, 2, 1)
    sns.boxplot(data=data,x='sport_activity',y='weight')
    plt.show()
    plt.subplot(2, 2, 1)
    sns.boxplot(data=data, x='sport_activity', y='sugar_levels')
    plt.show()

    plt.subplot(2, 2, 1)
    sns.histplot(data=data['num_of_siblings'])
    plt.show()
    print(data['num_of_siblings'].mean(), data['num_of_siblings'].median())
    print((data['num_of_siblings'].values))
    print((data['num_of_siblings'].values == 1).sum())
    print((data['num_of_siblings'].values == 2).sum())

# recieves dataframe
# returns the dataframe after cleaning the features from outliers
# and cleaning by bivaritae the following features - sugar levels and sport activity
# weight and age
def Q11(data,plot,is_training):
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


def Q14():
    pass
    #not coding question

def Q15():
    pass
    #not coding question

# imputes the feature "num of siblings" according to median as a strategy
def Q16(data):
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    data['num_of_siblings'] = imputer.fit_transform(data[['num_of_siblings']])
    print("meow")


# mean / median
# end of tail
# random sample
# constant value (arbitrary value)
# most frequent
def impute_simpleimputer(data,feature, impute_method):
    imputer = SimpleImputer(missing_values=np.nan, strategy=impute_method)
    data[feature] = imputer.fit_transform(data[[feature]])
    return data


def Q17(data,plot,is_training,save_csv,name_csv=""):
    median_impute = ['happiness_score','age', 'conversations_per_day', 'sport_activity', 'PCR_05', 'PCR_10']
    mean_impute = ['household_income', 'sugar_levels', 'PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_06', 'PCR_07', 'PCR_08', 'PCR_09']
    mf_impute = ['country']
    random_impute = ['pcr_date', 'home_country','num_of_siblings']
    const_impute = ['postcode']

    global IS_TRAINING
    global MEAN_MEDIAN_TRAINING
    global MOST_FR
    if is_training:
        IS_TRAINING = True
        relevent_data = data[median_impute + mean_impute]
        MEAN_MEDIAN_TRAINING = relevent_data.apply(meanMedian)
        for col in mf_impute:
            MOST_FR[col] = data[col].mode()[0]


        for i in median_impute:
            impute_simpleimputer(data, i, "median")

        for i in mean_impute:
            impute_simpleimputer(data, i, "mean")

        for i in mf_impute:
            impute_simpleimputer(data, i, "most_frequent")

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

        for col in mf_impute:
            imputer = SimpleImputer(missing_values=np.nan, strategy='constant',
                                    fill_value=MOST_FR[col])
            data[col] = imputer.fit_transform(data[[col]])


    if plot:
        print("before:")
        get_histograms(data,'weight')
        get_histograms(data,'sex')


    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    for i in const_impute:
        data[i] = imputer.fit_transform(data[[i]])

    imputer = RandomSampleImputer(random_state=my_id +or_id)
    for i in  random_impute:
        data[i] = imputer.fit_transform(data[[i]])

    #we use most frequent for sex
    # we use median for weight



    weight_list = prepare_weight_age_function(data)
    data['weight'] = data.apply( lambda  row: weight_list[int(row['age'])],axis=1)
    imputer = RandomSampleImputer(random_state=my_id + or_id)
    data['sex'] = imputer.fit_transform(data[['sex']])

    if plot:
        print("after:")
        get_histograms(data,'weight')
        get_histograms(data, 'sex')

    if save_csv:
        data.to_csv(f'{name_csv}.csv')

    return data

def Q19(data):
    #we need to show correlation to all others
    # for continuous
    for i in data.select_dtypes(include= np.number):
        print("Correlation between pcr_10 and  " + str(i) + " is: {:.3f}".format(data['PCR_10'].corr(data[i])))
    # for categorical
    print(data.groupby('sex')['PCR_10'].value_counts(normalize=True))


def Q20(data):
    COL_NAME = ['covid']
    COLS = 1
    ROWS = int(np.ceil(3 / COLS))
    for i, column in enumerate(COL_NAME, 1):
        ax = plt.subplot(1, 1, i)
        ax.title.set_text(column)
        sns.histplot(data=data, x='sport_activity', hue=column, kde=True)
        plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

def Q21(data):
    g = sns.jointplot(data.PCR_01,data.PCR_02,hue=data.risk)
    _ = g.ax_joint.grid()
    plt.show()


def Q22_plots(data,cols,target):

    data_pcr = data[cols+[target]]
    sns.pairplot(data_pcr, hue=target)
    plt.show()





def Q22(data):
    # plots_PCR(data=data)

    cols_1 = ['PCR_01','PCR_02','PCR_03','PCR_04','PCR_05','PCR_06'
                       ,'PCR_07', 'PCR_08','PCR_09','PCR_10','PCR_03_new']

    # # Q22_plots(data,cols_1,'risk')
    # cols_2 = ['age','weight','num_of_siblings','happiness_score','household_income','conversations_per_day'
    #                    ,'sugar_levels', 'sport_activity']
    #
    #
    # # Q22_plots(data, cols, 'covid')
    #
    # cols_3 = ['blood_type_A+','blood_type_A-','blood_type_AB+','blood_type_AB-','blood_type_B+','blood_type_B-'
    #                    ,'blood_type_O+', 'blood_type_O-','cough','fever','headache',
    #                 'low_appetite', 'shortness_of_breath'
    #           ]
    Q22_plots(data,cols_1, 'spread')

def Q23(data,save_csv=False,name_csv =''):
    remaining_features = [ 'age', 'sex', 'num_of_siblings', 'pcr_date', 'conversations_per_day',
                           'sugar_levels', 'sport_activity',  'PCR_01', 'PCR_02', 'PCR_03',
                          'PCR_04', 'PCR_05', 'PCR_06', 'PCR_08','PCR_09', 'PCR_10', 'blood_type_AB-', 'blood_type_B+',
                           'blood_type_B-','blood_type_O+', 'blood_type_O-','cough',
                           'fever','shortness_of_breath','country','postcode', 'is_army','home_country'
                           ]

    remaining_features_update = list()
    for i in remaining_features:
        if i in  data.columns.to_list():
            remaining_features_update.append(i)

    data = data[remaining_features_update]

    if save_csv:
        data.to_csv(f'{name_csv}.csv')

    return data

def prepare_weight_age_function(data):
    ret_list = []
    for i in range (0,100):
        tmp_data = data[ data['age'] == i ]
        ret_list.append(tmp_data['weight'].mean())
    return ret_list


def bi_variate(data,feature):
        for tmp in data:

            print((data.groupby(feature)[tmp].value_counts(normalize=True)))
            # pd.crosstab(data[feature], data[tmp])
            # x = pd.crosstab(data[feature], data[tmp]).plot(kind='bar', stacked=True)
            # print(x)
            # plt.show()

def remove_unwanted_features(data):
    toRemove = set()
    for i in data:
        if 'NAMED' in str(i).upper():
            toRemove.add(i)
    for i in toRemove:
        data.drop(i, axis=1, inplace=True)
    # data.drop('weight',axis=1,inplace=True)
    # data.drop('patient_id',axis=1,inplace=True)
    # data.drop('address', axis=1, inplace=True)
    # data.drop('current_location', axis=1, inplace=True)
    # data.drop('job', axis=1, inplace=True)
    # data.drop('symptoms', axis=1, inplace=True)
    # data.drop('county', axis=1, inplace=True)
    return data

def make_matshow(data):
    f = plt.figure()
    plt.matshow(data.corr(), fignum=f.number)

    show = ['age', 'sex', 'weight', 'num_of_siblings', 'happiness_score', 'household_income',
            'conversations_per_day',
            'sugar_levels', 'sport_activity', 'PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_05',
            'PCR_06', 'PCR_07', 'PCR_08', 'PCR_09', 'PCR_10', 'risk', 'spread', 'covid',
            'blood_type_A+', 'blood_type_A-', 'blood_type_AB+', 'blood_type_AB-',
            'blood_type_B+', 'blood_type_B-', 'blood_type_O+', 'blood_type_O-',
            'cough', 'fever', 'headache']
    plt.xticks(range(len(show)), show, fontsize=5,rotation=90)
    plt.yticks(range(len(show)), show, fontsize=5)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=3)
    _ = plt.title('Correlation Matrix', fontsize=5)
    plt.show()



if __name__ == '__main__':
    pass
    # data = pd.read_csv("virus_data.csv")
    # get_all_hist(train_data)
    # train_data, test_data = Q5(data)
    # train_data = Q6(train_data)
    #train_data = Q7(train_data)
    # train_data = Q8(train_data)
    #train_data = Q8(data)



