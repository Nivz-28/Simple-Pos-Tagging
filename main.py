import numpy as np
import tensorflow as tf


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, Bidirectional,Flatten
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras import utils as np_utils
from gensim.models import KeyedVectors
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Activation
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization

path = "/content/ca01"

def load_corpus(path):
    infile = open(path)
    sentence_list =[]
    for lines in infile:
        line_tokens = []
        sentence = lines.split()
        for str in sentence:
            tokens = str.split("/")
            line_tokens.append((tokens[0],tokens[1]))
        sentence_list.append(line_tokens)
    return sentence_list
data = load_corpus(path)

#print(data[2])
#print("Tagged sentences: ", len(data))

X=[]
Y=[]
for sentence in data:
    X_sentence = []
    Y_sentence = []
    for entity in sentence:
        X_sentence.append(entity[0]) # entity[0] contains the word
        Y_sentence.append(entity[1]) # entity[1] contains corresponding tag
        X.append(X_sentence)
        Y.append(Y_sentence)
#print('sample X: ', X[0], '\n')
#print('sample Y: ', Y[0], '\n')

words, tags = set([]), set([])

for s in X:
    for w in s:
        words.add(w.lower())
for ts in Y:
    for t in ts:
        tags.add(t)
word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs

tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding
#print(len(word2index))
E = len(word2index)

word_tokenizer = Tokenizer()              # instantiate tokeniser
word_tokenizer.fit_on_texts(X)            # fit tokeniser on data
# use the tokeniser to encode input sequence
train_X = word_tokenizer.texts_to_sequences(X)

tag_tokenizer = Tokenizer()
tag_tokenizer.fit_on_texts(Y)
train_Y = tag_tokenizer.texts_to_sequences(Y)

train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

for s in X:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    train_sentences_X.append(s_int)
for s in Y:
    train_tags_y.append([tag2index[t] for t in s])
#print(train_sentences_X[0])
#print(train_tags_y[0])

MAX_LENGTH = len(max(train_sentences_X, key=len))
#print(MAX_LENGTH)

train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
#print(train_sentences_X,'\n')
#print(train_tags_y)

from keras import backend as K

def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy

model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy', ignore_class_accuracy(0)])

model.summary()

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))

model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=30, validation_split=0.2)

num_test_sentences = int(input("Enter the number of test sentences: "))
test_samples = []
for _ in range(num_test_sentences):
    test_sentence = input("Enter a test sentence: ").split()
    test_samples.append(test_sentence)

# Preprocess user-input sentences
test_samples_X = []
for s in test_samples:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])
    test_samples_X.append(s_int)
test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')

# Use the model to predict tags for user-input sentences
predictions = model.predict(test_samples_X)

def logits_to_tokens(predictions, tag_index_mapping):
    token_predictions = []
    for pred_sequence in predictions:
        token_sequence = []
        for pred_logits in pred_sequence:
            pred_tag_index = int(np.argmax(pred_logits))
            pred_tag = tag_index_mapping[pred_tag_index]
            token_sequence.append(pred_tag)
        token_predictions.append(token_sequence)
    return token_predictions

# Convert predictions to token sequences
tag_index_mapping = {i: t for t, i in tag2index.items()}
token_predictions = logits_to_tokens(predictions, tag_index_mapping)

# Print token predictions
for i, tokens in enumerate(token_predictions):
    print(f"Predicted tags for test sentence {i + 1}:")
    print(tokens)

