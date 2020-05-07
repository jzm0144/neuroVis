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
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt


import seaborn as sb
from math import *
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
from scipy.stats import norm



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


trainModel(input_shape
          ,xTrain, yTrain
          ,xTest, yTest
          ,args.ageMatchUnmatch
          ,args.dataset
          ,num_classes
          ,batch_size=None
          ,epochs=100)

model = load_model('Models/'+ args.ageMatchUnmatch+"_"+args.dataset+'.h5')


'''
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


# ------------------------------  Part 3  ------------------------------------
# Delete the Examples where the Predictions were incorrect
m = []
for __ in range(inputs.shape[0]):
    predNeuron     =np.argmax(preds[__]),
    actualNeuron   =np.argmax(outs[__])
    if predNeuron != actualNeuron:
        m.append(__)
        print("skipped = ", __)

inputs = np.delete(inputs, m, 0)
preds  = np.delete(preds, m, 0)
outs   = np.delete(outs, m, 0)

# Balance the Class Instances
m = [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
inputs = np.delete(inputs, m, 0)
preds  = np.delete(preds, m, 0)
outs   = np.delete(outs, m, 0)
actualExplanations  = np.zeros(np.shape(inputs))
print(m); del m

actualLabels = np.array([np.argmax(outs[i]) for i in range(outs.shape[0])])

# Generation Permuations of the Labels
num_permutations = 1000
permutedLabels   = np.zeros((num_permutations,actualLabels.shape[0]), dtype=int)

for _ in range(num_permutations):

    temp = actualLabels.copy()
    np.random.shuffle(temp)
    permutedLabels[_, :] = temp.copy()


# For Now working with only Gradient Heatmap
heatmap = heatmaps[0]

#Create the analyzers
analyzer = innvestigate.create_analyzer(heatmap[0],
                                        model_no_softmax,
                                        neuron_selection_mode = "index",
                                        **heatmap[1])

# Generate the heatmaps

allExplanations = np.zeros((1+num_permutations, inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]))
for _ in range(actualLabels.shape[0]):
    label = actualLabels[_]
    actualExplanations[_] = analyzer.analyze(np.array([inputs[_]]), neuron_selection = label)
    #print(_, label)
allExplanations[0] = actualExplanations


permutedExplanations = np.zeros((num_permutations, inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]))

for permIndex in range(num_permutations):
    for _ in range(actualLabels.shape[0]):
        label = int(permutedLabels[permIndex, _])
        permutedExplanations[permIndex, _, :, :, :] = analyzer.analyze(np.array([inputs[_]]), neuron_selection = label)
        #print(_, label)
    if permIndex%50 == 0: print('Permutations Explanations Generated = ', permIndex)
allExplanations[1:] = permutedExplanations



# Calculate the P-Values and Fill them in the pMatrix
pMatrix = np.ones(inputs.shape[:])
for subj in range(inputs.shape[0]):
    for row in range(inputs.shape[1]):
        for col in range(inputs.shape[2]):
            vector = allExplanations[:, subj, row, col,0].copy()

            x = actualExplanations[subj, row, col]
            ipdb.set_trace()

            if x > 0:
                vector[vector < x] = 0
                vector[vector >= x] = 1
                p = np.sum(vector)/vector.shape[0]
            

            if x < 0:
                vector[vector > x] = 0
                vector[vector <= x] = 1
                p = np.sum(vector)/vector.shape[0]

            if x == 0:
                p == 1             # this case never occurs though

            pMatrix[subj,row, col, 0] = p
    if subj%5 == 0: print('Sujects = ', subj)



sb.heatmap(allExplanations[0,8,:,:,0],
           vmin = np.min(allExplanations[i,subj,:,:,0]),
           vmax = np.max(allExplanations[i,subj,:,:,0]),
           cmap ='RdBu')
plt.show()

plt.subplot(281)
sb.heatmap(pMatrix[0,:,:,0],
           vmin = 0,
           vmax = 1)
plt.subplot(282)
sb.heatmap(pMatrix[1,:,:,0],
           vmin = 0,
           vmax = 1)
plt.subplot(283)
sb.heatmap(pMatrix[2,:,:,0],
           vmin = 0,
           vmax = 1)
plt.subplot(284)
sb.heatmap(pMatrix[3,:,:,0],
           vmin = 0,
           vmax = 1)
plt.subplot(285)
sb.heatmap(pMatrix[8,:,:,0],
           vmin = 0,
           vmax = 1)
plt.subplot(286)
sb.heatmap(pMatrix[9,:,:,0],
           vmin = 0,
           vmax = 1)
plt.subplot(287)
sb.heatmap(pMatrix[10,:,:,0],
           vmin = 0,
           vmax = 1)
plt.subplot(288)
sb.heatmap(pMatrix[11,:,:,0],
           vmin = 0,
           vmax = 1)
plt.show()

plt.subplot(281)
plt.hist(pMatrix[0,:,:,0], bins = 10)
plt.subplot(282)
plt.hist(pMatrix[1,:,:,0], bins = 10)
plt.subplot(283)
plt.hist(pMatrix[2,:,:,0], bins = 10)
plt.subplot(284)
plt.hist(pMatrix[3,:,:,0], bins = 10)
plt.subplot(285)
plt.hist(pMatrix[8,:,:,0], bins = 10)
plt.subplot(286)
plt.hist(pMatrix[9,:,:,0], bins = 10)
plt.subplot(287)
plt.hist(pMatrix[10,:,:,0], bins = 10)
plt.subplot(288)
plt.hist(pMatrix[11,:,:,0], bins = 10)
plt.show()








i = 0;
subj1 = 0
subj2 = 1
subj15 = 2
subj16 = 4

plt.Figure(figsize=(7,7))
plt.subplot(221)
plt.title("All Exp, Subj1 "+str(np.max(allExplanations[0, subj1, 10, 10, 0])))
plt.plot(allExplanations[:, subj1, 10, 10, 0])

plt.subplot(222)
plt.title("All Exp, Subj2 "+str(np.max(allExplanations[0, subj2, 10, 10, 0])))
plt.plot(allExplanations[:, subj2, 10, 10, 0])

plt.subplot(223)
plt.title("All Exp, Subj15 "+str(np.max(allExplanations[0, subj15, 10, 10, 0])))
plt.plot(allExplanations[:, subj15, 10, 10, 0])

plt.subplot(224)
plt.title("All Exp, Subj16 "+str(np.max(allExplanations[0, subj16, 10, 10, 0])))
plt.plot(allExplanations[:, subj16, 10, 10, 0])

plt.show()




plt.Figure(figsize=(7,7))
plt.subplot(221)
plt.title("All Exp, Subj1 "+str(np.max(allExplanations[0, subj1, 10, 10, 0])))
plt.hist(allExplanations[:, subj1, 10, 10, 0])

plt.subplot(222)
plt.title("All Exp, Subj2 "+str(np.max(allExplanations[0, subj2, 10, 10, 0])))
plt.hist(allExplanations[:, subj2, 10, 10, 0])

plt.subplot(223)
plt.title("All Exp, Subj15 "+str(np.max(allExplanations[0, subj15, 10, 10, 0])))
plt.hist(allExplanations[:, subj15, 10, 10, 0])

plt.subplot(224)
plt.title("All Exp, Subj16 "+str(np.max(allExplanations[0, subj16, 10, 10, 0])))
plt.hist(allExplanations[:, subj16, 10, 10, 0])
plt.show()

'''







