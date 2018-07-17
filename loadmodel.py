import keras
import numpy as np
import random
import sys
import json
from keras.models import load_model
from keras.models import model_from_json
from keras.backend import manual_variable_initialization


from numpy.random import seed
seed(1)


model = load_model('C:\\Users\\chimi\\Desktop\\todo\\my_model.h5')
weights = model.get_weights()
#model.set_weights(weights)

text = open('C:\\Users\\chimi\\Desktop\\120TaylorSongsLyrics.txt',encoding='utf8').read().lower()
chars = sorted(list(set(text)))
char_indices = dict((char, chars.index(char)) for char in chars)
#print('los chars:', chars)
#print('los char indices:', char_indices)

#char_indices = (open('C:\\Users\\chimi\\Desktop\\todo\\char_indices.txt', 'r').read())
#with open('C:\\Users\\chimi\\Desktop\\todo\\chars.txt') as f:
#    chars = [list(literal_eval(line)) for line in f]


#chars= (open('C:\\Users\\chimi\\Desktop\\todo\\chars.txt', 'r').read())

#
#chars = sorted(list(set(text)))


#model.load_weights('C:\\Users\\chimi\\Desktop\my_model_weights.h5')
#with open('C:\\Users\\chimi\\Desktop\\model.json', 'r') as f:
#    model = model_from_json(f.read())

maxlen = 20
# We sample a new sequence every `step` characters
step = 3
# This holds our extracted sequences
sentences = []
# This holds the targets (the follow-up characters)
next_chars = []
def sample(preds, temperature=0.5):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)






texti = "ain't for the best m"
start_index = random.randint(0, len(text) - maxlen - 1)
#generated_text = text[5: 5 + maxlen]
generated_text = texti
print('--- Generating with seed: "' + generated_text + '"')
temperature = 0.5
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
