import numpy as np
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

file = open('f12.txt').read()

import nltk
nltk.download('stopwords')

def tokenize_words(input):
  input=input.lower()
  tokenizer=RegexpTokenizer(r'\w+')
  tokens=tokenizer.tokenize(input)
  filtered=filter(lambda token:token not in stopwords.words('english'),tokens)
  return "".join(filtered)
processed_inputs=tokenize_words(file)

chars= sorted(list(set(processed_inputs)))
char_to_num= dict((c,i) for i, c in enumerate(chars))

input_len = len(processed_inputs)
vocab_len= len(chars)
print('total no of chars:', input_len)
print('total vocab:', vocab_len)

seq_len = 100
x_data= []
y_data =[]
for i in range (0, input_len - seq_len,1):
    in_seq = processed_inputs[i:i + seq_len]
    out_seq = processed_inputs[i+seq_len]
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])
    
n_patterns = len(x_data)
print('total pat:', n_patterns)

X= np.reshape(x_data, (n_patterns, seq_len, 1))
X = X/float(vocab_len)
y=np_utils.to_categorical(y_data)

model = Sequential()
model.add(LSTM(256, input_shape= (X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences= True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation = 'softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam')
filepath='model_weights_saved.hdf5'
checkpoint=ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
desired_callbacks=[checkpoint]

model.fit(X,y, epochs=4, batch_size = 256, callbacks=desired_callbacks)

filename= 'model_weights_saved.hdf5'
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
num_to_char= dict((i,c) for i,c in enumerate(chars))

start=  np.random.randint(0,len(x_data)-1)
pattern= x_data[start]
print('Random Seed:')
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

for i in range(1000):
    x= np.reshape(pattern, (1,len(pattern), 1))
    x = x/float(vocab_len)
    prediction = model.predict(x,verbose=0)
    index= np.argmax(prediction)
    result = num_to_char[index]
    seq_in =[num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern= pattern[1:len(pattern)]
