from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential,Model
from keras.layers import LSTM, Flatten, Dense, Bidirectional,Input,Dropout

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers,regularizers, constraints

class Model:

    def __init__(self):
        
        pass

    def create_model(self ):

        model = Sequential()
        model.add(Bidirectional(LSTM(10, return_sequences = True),input_shape = (SEQ_LEN,300)))
        model.add(Bidirectional(LSTM(4)))
        model.add(Dense(8,activation = 'relu'))
        model.add(Dense(2,activation = 'softmax'))
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        return model