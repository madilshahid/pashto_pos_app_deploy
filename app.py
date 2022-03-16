import tensorflow as tf
from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from flask import Flask, request, render_template

with tf.device("cpu:0"):
    model = load_model('checkpoints/model.h5')

with open('data/data.pkl', 'rb') as f:
    X_train, Y_train, word2int, int2word, tag2int, int2tag = pickle.load(f)
    pred_tag = Y_train
    print(pred_tag)
    del X_train
    del Y_train


# flask
app = Flask(__name__)

@app.route("/", methods=["GET"])
def predict():
    return render_template('index.html')

@app.route("/prediction", methods=["POST"])
def prediction():
    sentence = request.form['Pashto'].split()
    # sentence = unicode(sentence, 'utf-8')
    abc = request.form['Pashto']

    tokenized_sentence = []

    for word in sentence:
        tokenized_sentence.append(word2int[word])

    dates = {}
    i = 0
    for token in tokenized_sentence:
        dates[sentence[i]] =  int2tag[pred_tag[token][0]]
        i = i + 1

#with tf.device("cpu:0"):
    #my_prediction = model.predict(padded_tokenized_sentence)



    #int2tag[0] = 'Unknown'
    #check = True
    #y = 49
    #z = len(sentence) - 1
    #while check:
       # b = np.argmax(my_prediction[0][y])

        #dates = {sentence[z] : int2tag[b]}
        #dates[sentence[z]] = int2tag[b]
        #y = y - 1
        #z = z - 1
        #if (z < 0):
         #   check = False


    return render_template('prediction.html', dates = dates , prediction_text = abc )

# instantiate flask


if __name__ == "__main__":
    app.run(debug=True)
