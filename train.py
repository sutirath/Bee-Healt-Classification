# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:05:48 2020

@author: Bqasx
"""

# Load various imports 
import os
import librosa
import librosa.display
from tensorflow import keras 
import numpy as np
import librosa
from pydub import AudioSegment
import matplotlib.pyplot as plt



path = './daataset2/'
X = []
Y = []
names = ['bad','enemy','good','queen']

for fn in os.listdir(path):
    if fn.endswith('.wav'):
        audio = AudioSegment.from_file(path+fn,format=None)
        x = audio
        x = np.array(x.get_array_of_samples(), np.float32)
        x = librosa.feature.mfcc(x)
        X.append(x)
        print(fn)
        print(x.shape)
        if fn[0]=='g':
            Y.append(names.index('good'))
        elif fn[0]=='b':
            Y.append(names.index('bad'))
        elif fn[0]=='e':
            Y.append(names.index('enemy'))
        else:
             Y.append(names.index('queen'))

X = np.array(X)
Y = np.array(Y)
X.shape

while True:
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, 5, input_shape=(20, 939, 1), activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.MaxPool2D())
    
    model.add(keras.layers.Conv2D(64, 5, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.MaxPool2D())
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(4, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
    history=model.fit(X[:,:,:,None], Y, epochs=20)
    if history.history['accuracy'][len(history.history['accuracy'])-1] >0.8:
        break
    
   
    

plt.plot(history.history['accuracy'][2:20])
plt.plot(history.history['loss'][2:20])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()

#predict
A = AudioSegment.from_file('test/queen_4.wav')
test=A
test = np.array(test.get_array_of_samples(), np.float32)
test=librosa.feature.mfcc(test)
names[model.predict(test[None,:,:,None]).argmax()]



# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")


