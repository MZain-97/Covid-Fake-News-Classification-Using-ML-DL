from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences




def tokenizer(text):
    sentences = sent_tokenize(text)
    data =pd.DataFrame()
    data['Sentences'] =sentences 
    return sentences,data

def  preprocessing(sentences):
    ps =PorterStemmer()
    corpus = []

    for i in range(0, len(sentences)):
        review = re.sub('[^a-zA-Z]',' ',sentences[i])
        review = review.lower()
        review = review.split()
        review  =[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    return corpus
    
def count_vectorizer(corpus,cv):

    X = cv.transform(corpus) 
    return X

def prediction(X,data,model):
     prediction= model.predict(X)
     prediction1 =prediction.tolist()
     data['Class'] = prediction1
     return prediction, data

def color_Fake_red(value):


            if value == 'fake' or value =='[Fake]' or value == 'Fake':
               color = 'red'
            else:
               color = 'Green'

            return 'color: %s' % color

def deep_preprocessing(corpus):
    voc_size=5000
    onehot_repr=[one_hot(words,voc_size)for words in corpus]
    sent_length=50
    embedded_docs=pad_sequences(onehot_repr,padding='post',maxlen=sent_length)
    X=np.array(embedded_docs)
    return X
def deep_prediction(X,data,model):
    prediction = model.predict(X)
    prediction= np.where(prediction>0.5,1,0)
    return prediction

def pred_result(prediction,data):
    prediction = np.where(prediction>0.5,"Real","Fake")
    data['Class'] = prediction
    return data