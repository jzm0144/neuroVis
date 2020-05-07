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

from statsmodels import stats




parser = argparse.ArgumentParser()
parser.add_argument("ageMatchUnmatch", type=str, help = "Enter ageMatched or ageUnmatched")
parser.add_argument("dataset", type=str, help="Dataset you want to train on \nADNI, ABIDE, ADHD or PTSD?")
parser.add_argument("heatmapNumber", type=int, help="Generate heatmap for the exmaple id")
parser.add_argument("topPaths", type=int, help="Most significant Paths")
parser.add_argument("label", type=int, help="Generate heatmap for what class? ")
parser.add_argument("methodNumber", type=int, help="Generate Explanations for Which Method?")
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
          ,num_classes
          ,batch_size=16
          ,epochs=50)
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
      #("lrp.sequential_preset_b_flat",  {"epsilon": 1}                     ,  "LRP-PresetBFlat"),

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
# At this point we have 9 Controls Subjects and 24 PTSD Subjects, these are the subjects where the classifier gave correct results

# Balance the Class Instances
m = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
inputs = np.delete(inputs, m, 0)
preds  = np.delete(preds, m, 0)
outs   = np.delete(outs, m, 0)
actualExplanations  = np.zeros(np.shape(inputs))
print(m); del m
# At this point, we have picked up 9 Controls and 9 PTSD subjects for apply a permuation test.

actualLabels = np.array([np.argmax(outs[i]) for i in range(outs.shape[0])])
actualIndices = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 17])
# At this Point we have labels like [c c c c c c c c c p p p p p p p p p]



# Test Statistic is the the Sum of heatscores for the Group1 (Labels Heatmaps in the list)

# Generation Permuations of the Indices
num_permutations = 1000
permutedIndices  = np.zeros((num_permutations,actualLabels.shape[0]), dtype=int)
totalData = np.zeros((num_permutations+1, actualLabels.shape[0]), dtype=float)

for _ in range(num_permutations):

    temp = actualIndices.copy()
    np.random.shuffle(temp)
    permutedIndices[_, :] = temp.copy()

# For Now working with only Gradient Heatmap
heatmap = heatmaps[args.methodNumber]#heatmaps[0]

#Create the analyzers
analyzer = innvestigate.create_analyzer(heatmap[0],
                                        model_no_softmax,
                                        neuron_selection_mode = "index",
                                        **heatmap[1])

# Generate the heatmaps
for _ in range(actualLabels.shape[0]):
    label = actualLabels[_]
    actualExplanations[_] = analyzer.analyze(np.array([inputs[_]]), neuron_selection = args.label) #np.argmax(outs[_])
    #print(_, label)

_, Row, Col, __ = actualExplanations.shape[:]

pMatrix = np.float64(np.zeros((Row, Col)))


for row in range(Row):
    for col in range(Col):
        thisElement = actualExplanations[:, row, col, 0]
        
        totalData[0,:] = thisElement

        for _ in range(num_permutations):
            i = _+1
            for j in range(actualLabels.shape[0]):
                totalData[i,j] = thisElement[permutedIndices[_,j]]
        
        pSums = []
        cSums = []

        tFlags = []

        for _ in range(totalData.shape[0]):

            pSums.append(np.sum(totalData[_,:9]))
            cSums.append(np.sum(totalData[_,9:]))

            pSums0 = pSums[0]
            cSums0 = cSums[0]

            if cSums0 > pSums0:
                tFlags.append(1) if (cSums[_] >= cSums[0]) else tFlags.append(0)
            if pSums0 > cSums0:
                tFlags.append(1) if (pSums[_] >= pSums[0]) else tFlags.append(0)
            if cSums0 == pSums0:
                print("Weird Thing Occurred")
                pVal = 0.5
        if len(tFlags) > 0:
            pVal = sum(tFlags)/len(tFlags)

        pMatrix[row, col] = pVal
        print("pValue = ", pVal)


# Benjamini Hochberg Procedure
RANK = pMatrix.copy()    # Get the sorting indices of the p Values
RANK = RANK.flatten()
RANK = RANK.argsort()
FDR = 0.05               # Set a FDR of 5%
numTests = 677



RANK = (RANK*FDR)/677

RANK = RANK.reshape(27, 27)

pMatrix[pMatrix > RANK] = 1 

########################################

pMatrix = pMatrix * (-1) + 1

pMatrix[pMatrix < 0.95] = 0.   # threshold 0.001



#sb.heatmap(pMatrix, vmin=0, vmax=1); plt.show()

# Save the Matrix of FDR Corrected p-values
df = pd.DataFrame(pMatrix)
df.to_csv(r'Results/'+str(args.methodNumber)+'.csv', index=False)

ipdb.set_trace()

# Save the Mean Heatmap Edge files
edge = saveEdgeFile(img = pMatrix,
                    idx = idx,
                    heatmap_method = heatmap[0],
                    clampNeuron = str(args.label),
                    topPaths = args.topPaths,
                    dataset = args.dataset,
                    xPath = xPath,
                    yPath = yPath,
                    map = "pos",
                    edgeDir = "Edge/")

# Also save the BrainNet png files
plotBrainNet(nodePath = "Node2/"+nodeFile,
             edgePath = edge,
             outputPath = 'Results/',
             configFile = 'config2.mat')