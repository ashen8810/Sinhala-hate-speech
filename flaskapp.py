from flask import Flask
from flask import render_template,request,url_for,redirect

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

import tensorflow as tf
from tensorflow import keras
import re
import string
model = tf.keras.models.load_model("tf_lstmmodel4.h5")
import numpy as np
def runner(sen):
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

    from keras import preprocessing
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
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
    return predictions,valid_predict
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/', methods=["GET", "POST"])
def run():
    if request.method == "POST":
        data = request.form['uname']
        prediction, valid = runner(data)
        return render_template("index.html", pred = prediction[0],valid = valid[0][0])
    return redirect(url_for('index.html'))


import json

@app.route('/hs/<name>')
# ‘/’ URL is bound with hello_world() function.
def hello(name):
    print(name)
    pred,val = runner(name)
    send = {"pred":str(pred),"val":str(val[0])}
    return send



# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()