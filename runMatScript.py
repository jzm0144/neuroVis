# Run Matlab Script with Python
import warnings
warnings.simplefilter('ignore')

import argparse
import os, sys

import matlab.engine

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="Dataset you want to train on \nADNI, ABIDE, ADHD or PTSD?")
args = parser.parse_args()
print("The Dataset is  = ",args.dataset)

path = os.getcwd()

surfacePath = 'Surface/BrainMesh_ICBM152.nv'
nodePath    = 'Node/'+ args.dataset+ '.node'
edgePath    = 'Edge/'+ args.edge_ '.edge'
configFile  = 'config.mat'

eng = matlab.engine.start_matlab(path)

eng.BrainNet_MapCfg(surfacePath,
                    nodePath,
                    edgePath,
                    outputFile,
                    configFile)
eng.quit()