from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import random
import os
import chardet

app = Flask(__name__)

def find_encoding(fname):
    r_file = open(fname, 'rb').read()
    result = chardet.detect(r_file)
    charenc = result['encoding']
    return charenc


files = os.listdir('Corpus/')

# print('TOTAL NO. OF FILES ', len(files), '\n')

raw_corpus = ''

for file in files[0:len(files)]:
    textencoding = find_encoding('Corpus/' + file)
    with open('Corpus/' + file, 'r', encoding=textencoding) as f:
        raw_corpus = raw_corpus + '' + f.read()

corpus = raw_corpus.split('\n')

# print('CORPUS SIZE', len(corpus), '\n')

corpus = [i.split('/') for i in corpus]

df = pd.DataFrame(corpus, columns=['Tag', 'Word', 'Trash'])
df.drop("Trash", axis=1, inplace=True)

df['Word_Tag'] = df[['Word', 'Tag']].apply(tuple, axis=1)

random.seed(1234)
# Splitting into training and test sets
train_set, test_set = train_test_split(df.Word_Tag, train_size=0.80)

# Get length of training and test sets

train_set

# Getting list of tagged words in training set
train_tagged_words = train_set

# Get length of total tagged words in training set
train_tagged_words

tokens = [pair[0] for pair in train_tagged_words]

# vocabulary
V = set(tokens)

# number of pos tags in the training corpus
T = set([pair[1] for pair in train_tagged_words])

# Create numpy array of no of pos tags by total vocabulary
t = len(T)
v = len(V)
w_given_t = np.zeros((t, v))


# compute word given tag: Emission Probability
def word_given_tag(word, tag, train_bag=train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    count_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]
    count_w_given_tag = len(w_given_tag_list)

    return (count_w_given_tag, count_tag)


def t2_given_t1(t2, t1, train_bag=train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t == t1])
    count_t2_t1 = 0
    for index in range(len(tags) - 1):
        if tags[index] == t1 and tags[index + 1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


tags_matrix = np.zeros((len(T), len(T)), dtype='float32')
for i, t1 in enumerate(list(T)):
    for j, t2 in enumerate(list(T)):
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0] / t2_given_t1(t2, t1)[1]

# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns=list(T), index=list(T))


# Viterbi Heuristic
def Viterbi(words, train_bag=train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    for key, word in enumerate(words):
        # initialise list of probability column for a given observation
        p = []
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['Punctuation', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # compute emission and state probabilities
            emission_p = (word_given_tag(words[key], tag)[0]) / (word_given_tag(words[key], tag)[1])
            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)]
        state.append(state_max)
    return list(zip(words, state))


complete_corpus = df.Word_Tag


@app.route("/", methods=["GET"])
def predict():
    return render_template('index.html')


@app.route("/prediction", methods=["POST"])
def prediction():

    sentence = request.form['Pashto']
    abc = request.form['Pashto']
    sentence = sentence.split(' ')

    def not_from_corpus(sentence, unique_words):
        for i in range(len(sentence) - 1):
            for j, pair in enumerate(complete_corpus):
                if sentence[i] not in unique_words:
                    return True
                    return False

    test_run_base = test_set
    test_tagged_words = [pair[0] for pair in test_set]

    start = time.time()
    tagged_seq = Viterbi(sentence)
    end = time.time()
    difference = end - start
    if not_from_corpus(sentence, V):
        repeated_tags = ['Noun', 'proNoun', 'Verb', 'NULL', 'Adverb', 'Adjective']
        for i in range(len(sentence)):
            not_in_corpus = False
            if sentence[i] not in V:
                not_in_corpus = True
            if not_in_corpus:
                tagged_seq[i] = (tagged_seq[i][0], random.choice(repeated_tags))
    dates = {}
    i = 0
    for token in enumerate(tagged_seq):
        dates[sentence[i]]  = token[1][1]
        i = i + 1

    return render_template('prediction.html', dates=dates, prediction_text=abc)


if __name__ == "__main__":
    app.run(debug=True)
