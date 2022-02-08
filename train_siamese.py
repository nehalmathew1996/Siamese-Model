# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:40:26 2021

@author: Nehal
"""


from siamese_model import build_siamese_model
import utils

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.datasets import mnist
import numpy as np

(trainx,trainy),(testx,testy) = mnist.load_data()
trainx = trainx/255.
testx = testx/255.

trainx=np.expand_dims(trainx,axis=-1)
testx=np.expand_dims(testx,axis=-1)

(pairTrain,labelTrain)=utils.make_pairs(trainx, trainy)
pairTrain.shape

(pairTest,labelTest)=utils.make_pairs(testx, testy)

imgA=Input(shape=(28,28,1))
imgB=Input(shape=(28,28,1))

feat_extrain=build_siamese_model((28,28,1))
featA=feat_extrain(imgA)
featB=feat_extrain(imgB)

print(featA,featB)

distance=Lambda(utils.euclidean_distance)([featA,featB])
outputs=Dense(1,activation='sigmoid')(distance)

model=Model(inputs=[imgA,imgB],outputs=outputs)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit([pairTrain[:,0],pairTrain[:,1]],labelTrain[:],
          validation_data=([pairTest[:,0],pairTest[:,1]],labelTest),
          batch_size=1024,
          epochs=2)
