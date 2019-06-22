from __future__ import print_function
from gensim.models import KeyedVectors

class Embeddings:

    def __init__(self):

        pass

    def create_embeddings(self):

        en_model = KeyedVectors.load_word2vec_format('data/cc.hi.300.vec')

        emb = {}

        for word in en_model.vocab:
            emb[word] = en_model[word]

        return emb