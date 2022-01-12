
import pandas as pd
from sklearn.model_selection import train_test_split



my_id = 3
or_id = 0
if __name__ == '__main__':
    # read the data set
    df = pd.read_csv('virus_labeled.csv')
    train, test = train_test_split(df, train_size=0.8, random_state=my_id + or_id)

    #


