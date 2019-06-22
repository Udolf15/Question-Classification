import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import re
import math

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

class Preprocessing:

    def __init__(self):

        pass

    def dfPreprocess( self, df ):

        print("Cleaning the training dataset ..........................")

        df = pd.read_csv("../input/questions/questions.csv")
        df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6'],axis = 1)
        df = df.dropna()
        
        print("Creating One hot encoding for the dataset ......................")
        df_temp = pd.get_dummies(df['Category'])
        df['Politics'] = df_temp[2.0]
        df['Movies'] = df_temp[3.0]
        df = df.drop(['Category'],axis = 1)

        df = shuffle(df)

        print("Dataset Cleaned and Ready for further processing")
        return df 

    def StopWords (self, string):
        
        print("Creating the list of Stop words..................")
        return string.split(' ')

    def text_to_array(self, text, SEQ_LEN, stopWords, emb):

        # Creates embedding array from the text array
        empyt_emb = np.zeros(300)
        text = self.basic_tokenizer(text[:-1], stopWords)[:SEQ_LEN]
        embeds = [emb.get(x, empyt_emb) for x in text]
        embeds += [empyt_emb] * (SEQ_LEN - len(embeds))
        arr =  np.array(embeds)

        return arr

    def basic_tokenizer(self, sentence, stopWords):

        # Tokenize the words and remove the stopwords from the list of words
        words = []
        for space_separated_fragment in sentence.strip().split():
            words.extend(_WORD_SPLIT.split(space_separated_fragment))
            
        return [w.lower() for w in words if w not in stopWords and w != '' and w != ' ']

    
    def batch_gen(self, df, batch_size,SEQ_LEN, stopWords, emb):
        
        print("Generating Training batch from training dataset")

        n_batches = math.ceil(len(df) / batch_size)
        while True: 
            df = df.sample(frac=1.)  # Shuffle the data.
            for i in range(n_batches):
                texts = df.iloc[i*batch_size:(i+1)*batch_size, 0]
                text_arr = np.array([self.text_to_array(text, SEQ_LEN, stopWords, emb) for text in texts])
                                yield text_arr, np.array([df["Politics"][i*batch_size:(i+1)*batch_size],df["Movies"][i*batch_size:(i+1)*batch_size]]).transpose()

    def batch_gen_test(self, texts, SEQ_LEN, stopWords, emb):

        print("Generating Testing batch from the input list of questions")
        
        text_arr = np.array([self.text_to_array(text,SEQ_LEN,stopWords,emb) for text in texts])
        yield text_arr