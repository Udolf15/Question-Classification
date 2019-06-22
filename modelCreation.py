from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential,Model
from keras.layers import LSTM, Flatten, Dense, Bidirectional,Input,Dropout

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers,regularizers, constraints
import math
from preprocessing import *

preProcessor = Preprocessing()

SEQ_LEN = 80

class ModelC:

    def __init__(self):
        
        pass

    def create_model(self ):
        
        print("Creating Model ......................")
        model = Sequential()
        model.add(Bidirectional(LSTM(10, return_sequences = True),input_shape = (SEQ_LEN,300)))
        model.add(Bidirectional(LSTM(4)))
        model.add(Dense(8,activation = 'relu'))
        model.add(Dense(2,activation = 'softmax'))
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        
        print("Model Created ")
        print(model.summary())

        return model

    def train(self, model, mg, train_df, batch_size, stopWords, emb):
        
        print("Starting the model training ........................")
        mg = preProcessor.batch_gen(train_df,batch_size,SEQ_LEN,stopWords,emb)
        model.fit_generator(mg, epochs=5,
                            steps_per_epoch=32,
                            verbose=True)

        print("Model Trained")

    def predict(self, model, mg, testQ, stopWords, emb):
        
        test_questions = preProcessor.batch_gen_test(testQ,SEQ_LEN,stopWords,emb)

        for ques in test_questions:
            pred = model.predict(ques)

        i = 0

        for prediction in pred:
            print("Question : ",testQ[i])
            i = i+1
            if(prediction[0] >= 0.5):
                print("Question belongs to Political Category")
                print("Confidence : ",prediction[0]*100,"%")
            else:
                print("Question belongs to Movies Category")
                print("Confidence : ",prediction[1]*100,"%")

    def save_model(self, model):

        print("Saving the model")
        model.save('model.h5')
        print("Model saved")