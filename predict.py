# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:38:29 2020

@author: Bqasx
"""


# load Model
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import numpy as np
import librosa
import librosa.display


names = ['bad','enemy','good','queen']
model = load_model('./model/model.h5')
model.summary()


#predict
A = AudioSegment.from_file('test/queen_2.wav')
test=A
test = np.array(test.get_array_of_samples(), np.float32)
test=librosa.feature.mfcc(test)
sound = test[:,:939]
classi = names[model.predict(sound[None,:,:,None]).argmax()]
acc = model.predict(sound[None,:,:,None]).max()*100
print('Class = '+str(classi) +'  Accuracy = %.2f' %acc)



import matplotlib.pyplot as plt

CQT = librosa.amplitude_to_db(test, ref=np.max)
librosa.display.specshow(test, x_axis='time')
plt.subplot(4, 2, 4)
librosa.display.specshow(CQT, x_axis='time')
plt.colorbar()
plt.title('Tempogram')
plt.tight_layout()
plt.show()




