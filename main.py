"""
Created on Sunday Jan 9th 19:31:41 2020
@author: Janzaib Masood
"""
from __future__ import print_function
import warnings
warnings.simplefilter('ignore')
import argparse
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sb


from math import *
from itertools import combinations
import ipdb as ipdb
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
import innvestigate
from Transformer import Transformer
from utils import *
from models import trainModel


parser = argparse.ArgumentParser()
parser.add_argument("ageMatchUnmatch", type=str, help = "Enter ageMatched or ageUnmatched")
parser.add_argument("dataset", type=str, help="Dataset you want to train on \nADNI, ABIDE, ADHD or PTSD?")
parser.add_argument("heatmapNumber", type=int, help="Generate heatmap for the exmaple id")
parser.add_argument("topPaths", type=int, help="Most significant Paths")
parser.add_argument("label", type=int, help="Generate heatmap for what class? ")
args = parser.parse_args()


print("The Dataset is                    = ", args.dataset)
print("The age                           = ", args.ageMatchUnmatch)
print("Generate Heatmap for Data Example = ", args.heatmapNumber)
print("Num of Paths in Heatmap           = ", args.topPaths)
print("Clamped Neuron                    = ", args.label)

path = os.getcwd()
trainPath = os.getcwd()+ "/Data/" +args.ageMatchUnmatch+"/"+ args.dataset + "_train_data.xlsx"
testPath  = os.getcwd()+ "/Data/" +args.ageMatchUnmatch+"/"+ args.dataset + "_test_data.xlsx"
clf = Transformer(trainPath, testPath, verbose=True)
xTrain, yTrain = clf.getTrainData()
xTest, yTest   = clf.getTestData()
idx, xPath, yPath = clf.getPaths()
codeLabels(yTrain = yTrain, yTest = yTest, disorder = args.dataset)

# Bring all Data in range 0.0 and 1.0
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain = (xTrain + 1)/2
xTest  = (xTest  + 1)/2

num_classes = len(np.unique(yTrain))
print("Number of Classes", num_classes)


# Input Image Dimensions (The reshaped reduced Connectivity Matrix)
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

'''
trainModel(input_shape
          ,xTrain, yTrain
          ,xTest, yTest
          ,args.ageMatchUnmatch
          ,args.dataset
          ,num_classes)
'''

model = load_model('Models/'+ args.ageMatchUnmatch+"_"+args.dataset+'.h5')
# Strip softmax layer
model_no_softmax = innvestigate.utils.model_wo_softmax(model)



#Decide your inputs
inputs = xTest[:,:,:,:]
outs   = yTest[:]
preds = model.predict(inputs)
#y_yHat = []
for i in range(len(preds)):
    print("Out= ",outs[i], "  Pred   ",preds[i])
    #y_yHat.append(str(np.argmax(outs[i])) + str(np.argmax(preds[i])))
#print(y_yHat)


input_max = 1.0
input_min = 0.0
noise_scale = (input_max - input_min) * 0.005

heatmaps = [
      # NAME                                  OPT.PARAMS                     TITLE
      # Show input.

      # Gradient Family
      ("gradient",                      {}                                 ,  "Gradient"),
      ("smoothgrad",                    {"augment_by_n": 1,
                                         "noise_scale": noise_scale,
                                         "postprocess": "square"}          ,  "SmoothGrad"),
      ("input_t_gradient",              {}                                 ,  "Input * Gradient"),
      #("integrated_gradients",          {"reference_inputs": input_min,
      #                                   "steps": 16}                      ,  "Integrated Gradients"),

      ("deconvnet",                     {}                                 ,  "Deconvnet"),
      ("guided_backprop",               {}                                 ,  "Guided Backprop"),

      # LRP Family
      ("deep_taylor.bounded",           {"low": input_min,
                                         "high": input_max}                ,  "DeepTaylor"),
      ("lrp.z",                         {}                                 ,  "LRP-Z"),
      ("lrp.epsilon",                   {"epsilon": 1}                     ,  "LRP-Epsilon"),
      ("lrp.sequential_preset_a_flat",  {"epsilon": 1}                     ,  "LRP-PresetAFlat"),
      ("lrp.sequential_preset_b_flat",  {"epsilon": 1}                     ,  "LRP-PresetBFlat"),

      # State of the Art Methods
      #("occlusion",                     {}                                 ,  "Occlusion Map"),
      #("lime",                          {}                                 ,  "Lime Method"),
      #("shapley",                       {}                                 ,  "Shapely Values"),
      #("Meaningful Perturbation",       {}                                 ,  "Meaningful Perturbation")
]



# The Node File
temp = {"ADNI":200, "ABIDE":200,"ADHD":190, "PTSD":125}
nodeFile = args.dataset + str(temp[args.dataset]) + ".node"

