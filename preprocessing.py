import pandas as pd
import numpy as np
from sklearn.utils import shuffle

class Preprocessing:

    def __init__(self):

        pass

    def dfPreprocess( self, df ):

        df = pd.read_csv("../input/questions/questions.csv")
        df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6'],axis = 1)
        df = df.dropna()
        
        df_temp = pd.get_dummies(df['Category'])
        df['Politics'] = df_temp[2.0]
        df['Movies'] = df_temp[3.0]
        df = df.drop(['Category'],axis = 1)

        df = shuffle(df)

        return df 