import pandas as pd
from math import *
import numpy as np

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

    def getPaths(self):
        self.Q = pd.read_excel(self.testPath)
        idx = self.Q.iloc[1:,0].values
    
        paths  = self.Q.iloc[1:,1].values
        xID = []
        yID = []
        spotID = []
        for i in range(paths.shape[0]):
            spotID.append(int(idx[i]))
            xID.append(int(paths[i].split(',')[0][1:]))
            yID.append(int(paths[i].split(',')[1][:-1]))

        return (spotID, xID, yID)


