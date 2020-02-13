"""
Created on Sund Jan 9th 19:31:41 2020
@Author: Janzaib Masood
"""
from __future__ import print_function
import warnings
warnings.simplefilter('ignore')
import argparse
import os, sys
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
from math import * 
import ipdb as ipdb

import keras
import innvestigate
from Transformer import Transformer
from utils import *
from models import trainModel


parser = argparse.ArgumentParser()
parser.add_argument("ageMatchUnmatch", type=str, help= "Enter ageMatched or ageUnmatched")
parser.add_argument("dataset", type=str, help= "Dataset you want to train on \nADNI, ABIDE, ADHD, PTSD?")
parser.add_argument("heatmapNumber", type=int, help= "Generate Heatmap for this Example ID")
parser.add_argument("topPaths", type=int, help= "Most Significant Paths")
parser.add_argument("label", type=int, help="Generate Heatmap for what class \n Enter class ID")

args = parser.parse_args()


print("The Dataset is                                         = ", args.dataset)
print("The Age                                                = ", args.ageMatchUnmatch)
print("Generate Heatmap for Data Example                      = ", args.heatmapNumber)
print("Number of Paths in the Heatmap to be display           = ", args.topPaths)
print("Clamped Neuron                                         = ", args.label)


