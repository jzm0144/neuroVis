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


# ------------------------------  Measure of Dissimilarity between  ------------------------
# ------------------------------  PTSD and Control Clamping       ------------------------

for heatmap in heatmaps:
    #Create the analyzers
    analyzer = innvestigate.create_analyzer(heatmap[0],
                                            model_no_softmax,
                                            neuron_selection_mode = "index",
                                            **heatmap[1])

    # Generate the heatmaps
    analysis_ctrl_clamp = analyzer.analyze(inputs, 0)
    analysis_ptsd_clamp = analyzer.analyze(inputs, 1)

    # Delete the explanations where prediction was incorrect
    m = []
    for __ in range(analysis_ctrl_clamp.shape[0]):
        predNeuron=np.argmax(preds[__]),
        actualNeuron=np.argmax(outs[__])
        if predNeuron != actualNeuron:
            m.append(__)
            print("skipped = ", __)
    analysis_ctrl_clamp = np.delete(analysis_ctrl_clamp, m, 0)
    analysis_ptsd_clamp = np.delete(analysis_ptsd_clamp, m, 0)
    print(m)
    del m
    # Working with 8 Control Subjects
    controls_with_ctrl_clamp  = analysis_ctrl_clamp[:8,:,:,0]
    controls_with_ptsd_clamp  = analysis_ptsd_clamp[:8,:,:,0]



    
    groupItems = controls_with_ctrl_clamp.shape[0]

    comb8c = list(combinations([_ for _ in range(8)], 2))

    control_ctrl_clamp_Dist = 0
    control_ptsd_clamp_Dist = 0
    bwgroupsDist = 0
    for _ in comb8c:
        i, j = _

        control_ctrl_clamp_Dist  += np.sqrt(np.square(controls_with_ctrl_clamp[i] - controls_with_ctrl_clamp[j]))
        control_ptsd_clamp_Dist  += np.sqrt(np.square(controls_with_ptsd_clamp[i] - controls_with_ptsd_clamp[j]))
        bwgroupsDist             += np.sqrt(np.square(controls_with_ctrl_clamp[i] - controls_with_ptsd_clamp[j]))

    print("Distance Within Controls    = ", np.sum(control_ctrl_clamp_Dist/len(comb8c)))
    print("Distance Within PTSD        = ", np.sum(control_ptsd_clamp_Dist/len(comb8c)))
    print("Distance b/w PTSD & Control = ", np.sum(bwgroupsDist/len(comb8c)))
    
    plt.figure(1)
    plt.subplot(331)
    m1 = sb.heatmap(control_ctrl_clamp_Dist/len(comb8c))
    plt.subplot(335)
    m2 = sb.heatmap(control_ptsd_clamp_Dist/len(comb8c))
    plt.subplot(339)
    m3 = sb.heatmap(bwgroupsDist/len(comb8c))
    plt.savefig('dist_control.png')
    
    # Working with 8 PTSD Subjects
    ptsd_with_ctrl_clamp  = analysis_ctrl_clamp[8:16,:,:,0]
    ptsd_with_ptsd_clamp  = analysis_ptsd_clamp[8:16,:,:,0]

    
    groupItems = ptsd_with_ctrl_clamp.shape[0]

    comb8c = list(combinations([_ for _ in range(8)], 2))

    ptsd_ctrl_clamp_Dist = 0
    ptsd_ptsd_clamp_Dist = 0
    bwgroupsDist = 0
    for _ in comb8c:
        i, j = _

        ptsd_ctrl_clamp_Dist  += np.sqrt(np.square(ptsd_with_ctrl_clamp[i] - ptsd_with_ctrl_clamp[j]))
        ptsd_ptsd_clamp_Dist  += np.sqrt(np.square(ptsd_with_ptsd_clamp[i] - ptsd_with_ptsd_clamp[j]))
        bwgroupsDist          += np.sqrt(np.square(ptsd_with_ctrl_clamp[i] - ptsd_with_ptsd_clamp[j]))

    print("Distance Within Controls    = ", np.sum(ptsd_ctrl_clamp_Dist/len(comb8c)))
    print("Distance Within PTSD        = ", np.sum(ptsd_ptsd_clamp_Dist/len(comb8c)))
    print("Distance b/w PTSD & Control = ", np.sum(bwgroupsDist/len(comb8c)))

    plt.subplot(331)
    m1 = sb.heatmap(ptsd_ctrl_clamp_Dist/len(comb8c))
    plt.subplot(335)
    m2 = sb.heatmap(ptsd_ptsd_clamp_Dist/len(comb8c))
    plt.subplot(339)
    m3 = sb.heatmap(bwgroupsDist/len(comb8c))
    plt.savefig('dist_ptsd.png')

    ipdb.set_trace()
