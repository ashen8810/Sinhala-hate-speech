import streamlit as st 
st.title("Sinhala Hate Speech Detector")
sen = st.text_input("Phrase")
# sen = "පොන්නයා,1ab"
import tensorflow as tf
from tensorflow import keras
import re
import string
model = tf.keras.models.load_model("tf_lstmmodel2.h5")
import numpy as np
#
# def remove_punct(text):
#     translator = str.maketrans("", "", string.punctuation)
#     return text.translate(translator)
#
# pattern = re.compile(r"https?://(\S+|www)\.\S+")
#
#
#
# sent = remove_punct(sen)
# stop = set([";","'","...","වෙනි","ක","ද","ශ්\u200dරී","ට","ම","ක","' '","වෙනි"])
# def remove_stopwords(text):
#     filtered_words = [word for word in text.split() if word not in stop]
#     return " ".join(filtered_words)
# def remove_digits(text):
#     # url = re.compile(r"https?://\S+|www\.\S+")
#     output = re.sub(r'\s*[A-Za-z]+\b', '' , text)
#     return re.sub(r"(^|\W)\d+", "", output)
# dff= remove_digits(sent)
# dff = remove_stopwords(sent)
#

exclude = set(",.:;'\"-?!/´`%,..abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def removePunctuation(txt):
    return ''.join([(c if c not in exclude else " ") for c in txt])


def removeNumbers(txt):
    return ''.join(c for c in txt if not c.isnumeric())


def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


pattern = re.compile(r"https?://(\S+|www)\.\S+")

# sent = remove_punct(sen)

stop = set([";", "'", "...", "වෙනි", "ක", "ද", "ශ්\u200dරී", "ට", "ම", "ක", "' '", "වෙනි"])


def remove_stopwords(text):
    filtered_words = [word for word in text.split() if word not in stop]
    return " ".join(filtered_words)

from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
sen = remove_stopwords(sen)
# sen = remove_punct(sen)
sen = removeNumbers(sen)
sen = removePunctuation(sen)
X_t = [sen]
tokenizer = Tokenizer(50000)
tokenizer.fit_on_texts(X_t)
x_t = np.array(tokenizer.texts_to_sequences(X_t) )
x_t = pad_sequences(x_t, padding='post', maxlen=100)
# print(x_t)


valid_predict= model.predict(x_t)
# print(valid_predict)



# print(sen)

# predictions = model.predict([f])
predictions = [1 if p > 0.5 else 0 for p in valid_predict[0]]
print(predictions)
k = ""
if(predictions[0]==1):
    k = "Dont Use Bad Words"
else:
    k = "Good words"
# print(k)
st.text(valid_predict[0])
st.info(k)