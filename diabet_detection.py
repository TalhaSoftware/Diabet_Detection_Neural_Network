

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:15:25 2020

@author: Talha Yazilim
"""

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input
from keras.optimizers import SGD

from sklearn.impute import SimpleImputer

veriler = pd.read_csv("diabetes_data_upload.csv")

age = veriler.iloc[:,0]
X = veriler.iloc[:,1:16]
y = veriler.iloc[:,16:]


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

for i in range(16):
    try:
        le = LabelEncoder()
        X.iloc[:,i] = le.fit_transform(X.iloc[:,i])
    except IndexError:
        pass
    
x = pd.concat([age,X],axis=1)
y = y.reshape(-1,1)

model = Sequential()
model.add(Dense(128,input_dim=16))
model.add(Activation("relu"))
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dense(512))
model.add(Activation("softmax"))

model.compile(optimizer="sgd",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(x,y,batch_size=10,epochs=50,validation_split=0.13)


tahmin = model.predict_classes(np.array([[32,0,0,0,0,0,1,1,1,1,1,0,0,1,0,1]])).reshape(-1,1)
print(tahmin)








