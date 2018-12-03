import random
import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from keras import optimizers
import keras
from keras import layers

path='../data/review_sample_5.json'
reviews=[]
with open(path, 'r') as f:
    data = json.load(f)[:25000]
    for i in data:
        if(len(i["text"]) < 251):
            reviews.append(i["text"])

print ("Length of the reviews")
len(reviews)

filename = 'short_reviews_shuffle.txt'
pd.DataFrame(reviews).to_csv(filename, header=None, index=None, sep=' ')

text = open('short_reviews_shuffle.txt').read()
print('Corpus length:', len(text))

chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
# Dictionary mapping unique characters to their index in `chars`
char_indices = dict((char, chars.index(char)) for char in chars)

maxlen=60
step=1

def getDataFromChunk(txtChunk, maxlen=60, step=1):
    sentences = []
    next_chars = []
    for i in range(0, len(txtChunk) - maxlen, step):
        sentences.append(txtChunk[i : i + maxlen])
        next_chars.append(txtChunk[i + maxlen])
    print('nb sequences:', len(sentences))
    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
    return [X, y]

model = keras.models.Sequential()
model.add(layers.LSTM(1024, input_shape=(maxlen, len(chars)),return_sequences=True))
model.add(layers.LSTM(1024, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

filepath="Feb-22-all-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
              patience=1, min_lr=0.00001)
callbacks_list = [checkpoint, reduce_lr]

def sample(preds, temperature=1.0):
    '''
    Generate some randomness with the given preds
    which is a list of numbers, if the temperature
    is very small, it will always pick the index
    with highest pred value
    '''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if __name__ == "__main__":

    for iteration in range(1, 20):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        with open("short_reviews_shuffle.txt") as f:
            for chunk in iter(lambda: f.read(90000), ""):
                X, y = getDataFromChunk(chunk)
                model.fit(X, y, batch_size=128, epochs=1, callbacks=callbacks_list)

        # Select a text seed at random
        start_index = random.randint(0, len(text) - maxlen - 1)
        generated_text = text[start_index: start_index + maxlen]
        print('--- Generating with seed: "' + generated_text + '"')

        for temperature in [0.5, 0.8, 1.0]:
            print('------ temperature:', temperature)
            sys.stdout.write(generated_text)

            # We generate 300 characters
            for i in range(300):
                sampled = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(generated_text):
                    sampled[0, t, char_indices[char]] = 1.

                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_char = chars[next_index]

                generated_text += next_char
                generated_text = generated_text[1:]

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()