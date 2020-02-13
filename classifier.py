# -*- coding: utf-8 -*-
"""
Created on Sunday Nov 29 19:31:41 2019

@author: Janzaib Masood
"""



from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import type_of_target

import argparse
import os, sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from math import *
import ipdb as ipdb

parser = argparse.ArgumentParser()
parser.add_argument("ageMatchUnmatch", type=str, help = "Enter ageMatched or ageUnmatched")
parser.add_argument("dataset", type=str, help="Which Dataset you want to train on \nADNI, ABIDE, ADHD or PTSD?")
args = parser.parse_args()
print("The Dataset is  = ",args.dataset)
print("The age =  ",args.ageMatchUnmatch)

class Transformer:
    def __init__(self, trainPath, testPath, verbose):
        self.trainPath = trainPath
        self.testPath  = testPath
        self.verbose   = verbose
        self.xTrain    = None
        self.yTrain    = None
        self.xTest     = None
        self.yTest     = None
        self.numSubjects = None
        self.numPaths    = None
        self.picDim      = None
        self.picLength = None
        self.sqrLength = None
        self.vecLength = None
        self.o = None
        self.Q = None
        self.Subjects = None

    def vecFix(self, a):
        self.vecLength = a.shape[0]
        self.sqrLength = sqrt(a.shape[0])

        self.picLength = ceil(self.sqrLength)

        if self.sqrLength**2 == self.picLength**2:
            self.out = a
        else:
            b = np.zeros(self.picLength**2)
            b[:self.vecLength] = a[:]
            self.out = b
        return self.out

    def vec2Square(self, a):
        self.o = self.vecFix(a)
        return self.o.reshape((int(sqrt(self.o.shape[0])), int(sqrt(self.o.shape[0]))))

    def getTrainData(self):
        self.Q = pd.read_excel(self.trainPath)
        self.Subjects   = self.Q.iloc[1:,2:].values

        self.numSubjects = self.Subjects.shape[1]
        self.numPaths   = self.Subjects.shape[0]
        self.picDim = self.vec2Square(self.Subjects[:,0]).shape[0]

        if self.verbose == True:
          print("Training Data (Num of Subjects, Connectivity Paths)  = (", self.numSubjects, " ,",self.numPaths,")")

        self.xTrain = np.zeros((self.numSubjects, self.picDim, self.picDim))
        self.yTrain = self.Q.iloc[0,2:].values 
        for subjId in range(self.numSubjects):
            self.xTrain[subjId, :, :] = self.vec2Square(self.Subjects[:,subjId])
        return self.xTrain, self.yTrain

    def getTestData(self):
        self.Q = pd.read_excel(self.testPath)
        self.Subjects   = self.Q.iloc[1:,2:].values

        self.numSubjects = self.Subjects.shape[1]
        self.numPaths    = self.Subjects.shape[0]
        self.picDim = self.vec2Square(self.Subjects[:,0]).shape[0]

        if self.verbose == True:
          print("Testing Data (Num of Subjects, Connectivity Paths)  = (", self.numSubjects, " ,",self.numPaths,")")

        self.xTest = np.zeros((self.numSubjects, self.picDim, self.picDim))
        self.yTest = self.Q.iloc[0,2:].values 
        for subjId in range(self.numSubjects):
            self.xTest[subjId, :, :] = self.vec2Square(self.Subjects[:,subjId])
        return self.xTest, self.yTest

def codeLabels(disorder): #ABIDE, ADHD, PTSD, ADNI
    if disorder == "ADNI":
        yTrain[yTrain == 'Controls'] = 0
        yTrain[yTrain == 'EMCI']     = 1
        yTrain[yTrain == 'LMCI']     = 2
        yTrain[yTrain == 'AD']       = 3

        yTest[yTest   == 'Controls'] = 0
        yTest[yTest   == 'EMCI']     = 1
        yTest[yTest   == 'LMCI']     = 2
        yTest[yTest   == 'AD']       = 3
    if disorder == "ADHD":
        yTrain[yTrain == 'Controls'] = 0
        yTrain[yTrain == 'ADHD-C']   = 1
        yTrain[yTrain == 'ADHD-H']   = 2
        yTrain[yTrain == 'ADHD-I']   = 3

        yTest[yTest   == 'Controls'] = 0
        yTest[yTest   == 'ADHD-C']   = 1
        yTest[yTest   == 'ADHD-H']   = 2
        yTest[yTest   == 'ADHD-I']   = 3
    if disorder == "ABIDE":
        yTrain[yTrain == 'Controls'] = 0
        yTrain[yTrain == 'Aspergers']= 1
        yTrain[yTrain == 'Autism']   = 2

        yTest[yTest   == 'Controls'] = 0
        yTest[yTest   == 'Aspergers']= 1
        yTest[yTest   == 'Autism']   = 2
    if disorder == "PTSD":
        yTrain[yTrain == 'Controls'] = 0
        yTrain[yTrain == 'PCS_PTSD'] = 1
        yTrain[yTrain == 'PTSD']     = 1

        yTest[yTest   == 'Controls'] = 0
        yTest[yTest   == 'PCS_PTSD'] = 1
        yTest[yTest   == 'PTSD']     = 1


path = os.getcwd()
trainPath = os.getcwd()+ "/Data/" +args.ageMatchUnmatch+"/"+ args.dataset + "_train_data.xlsx"
testPath  = os.getcwd()+ "/Data/" +args.ageMatchUnmatch+"/"+ args.dataset + "_test_data.xlsx"

clf = Transformer(trainPath, testPath, verbose=True)
xTrain, yTrain = clf.getTrainData()
xTest, yTest   = clf.getTestData()

codeLabels(disorder = args.dataset)

# Brain all Data in range 0.0 and 1.0
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain = (xTrain + 1)/2
xTest  = (xTest  + 1)/2

batch_size = 64
num_classes = len(np.unique(yTrain))
print("len of unique entries", num_classes)
epochs = 64

# input image dimensions
img_rows, img_cols = xTrain.shape[1:]
print("Image Size = ("+str(img_rows)+", "+str(img_cols),")")


if K.image_data_format() == 'channels_first':
    xTrain = xTrain.reshape(xTrain.shape[0], 1, img_rows, img_cols)
    xTest  = xTest.reshape(xTest.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    xTrain = xTrain.reshape(xTrain.shape[0], img_rows, img_cols, 1)
    xTest  = xTest.reshape(xTest.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


#Accuracy
# convert class vectors to binary class matrices
yTrain = keras.utils.to_categorical(yTrain, num_classes)
yTest = keras.utils.to_categorical(yTest, num_classes)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(12, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(xTrain, yTrain,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(xTest, yTest))
score = model.evaluate(xTest, yTest, verbose=0)
model.save(args.ageMatchUnmatch+"_"+args.dataset+'.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])



