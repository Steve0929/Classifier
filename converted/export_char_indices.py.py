import keras
import numpy as np
import random
import sys
import json
text = open('C:\\Users\\chimi\\Desktop\\tay.txt').read().lower()
print('Text Corpus length:', len(text))

maxlen = 25
# We sample a new sequence every `step` characters
step = 3
# This holds our extracted sequences
sentences = []
# This holds the targets (the follow-up characters)
next_chars = []
#range(start_value, end_value, step)
for i in range(0, len(text)-maxlen , step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))
# List of unique characters in the corpus
chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
# Dictionary mapping unique characters to their index in `chars`
char_indices = dict((char, chars.index(char)) for char in chars)

with open('C:\\Users\\chimi\\Desktop\\file.txt', 'w') as file:
     file.write(json.dumps(char_indices))

# Next, one-hot encode the characters into binary arrays.
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


import tensorflowjs as tfjs
#tfjs.converters.save_keras_model(model, 'C:\\Users\\chimi\\Desktop')
