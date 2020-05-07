import argparse
import pandas as pd
import numpy as np
import matlab.engine
import os, sys
import ipdb as ipdb
import bottleneck as bn

def codeLabels(yTrain, yTest, disorder): #ABIDE, ADHD, PTSD, ADNI
    if disorder == "ADNI":
        yTrain[yTrain == 'Controls'] = 0
        yTrain[yTrain == 'EMCI']     = 1
        yTrain[yTrain == 'LMCI']     = 1
        yTrain[yTrain == 'AD']       = 1

        yTest[yTest   == 'Controls'] = 0
        yTest[yTest   == 'EMCI']     = 1
        yTest[yTest   == 'LMCI']     = 1
        yTest[yTest   == 'AD']       = 1
    if disorder == "ADHD":
        yTrain[yTrain == 'Controls'] = 0
        yTrain[yTrain == 'ADHD-C']   = 1
        yTrain[yTrain == 'ADHD-H']   = 1
        yTrain[yTrain == 'ADHD-I']   = 1

        yTest[yTest   == 'Controls'] = 0
        yTest[yTest   == 'ADHD-C']   = 1
        yTest[yTest   == 'ADHD-H']   = 1
        yTest[yTest   == 'ADHD-I']   = 1
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



def square2Vec(img, vecLength):
    imgVec = img.flatten()
    vec = imgVec[:vecLength]
    return vec


def vector2Conn(indices, xPath, yPath, disorder, strengthVector):
    matSize = {"ADNI":200, "ABIDE":200,"ADHD":190, "PTSD":125}
    
    M = np.zeros((matSize[disorder], matSize[disorder]))
    for __ in indices:
        index = __ - 1

        M[xPath[index]-1, yPath[index]-1] = strengthVector[index]
    return M


def passTopPaths(connVec, top):   
    length = connVec.shape[0]

    temp = np.zeros(connVec.shape)
    indices = np.argsort(connVec)

    for i in range(length-top, length):
        ind = indices[i]
        temp[ind] = connVec[ind]

    for i in range(top):
        ind = indices[i]
        temp[ind] = connVec[ind]

    return temp


def saveEdgeFile(img,
                 idx,
                 heatmap_method,
                 clampNeuron,
                 topPaths,
                 dataset,
                 xPath,
                 yPath,
                 predNeuron="",
                 actualNeuron="",
                 map = "all",
                 edgeDir = "Edge/",
                 exampleHNum=""):

    thisEdge  = ''

    vec = square2Vec(img, vecLength=len(idx))

    vec = passTopPaths(vec, top = topPaths)


    conn = vector2Conn(idx, xPath, yPath, disorder = dataset, strengthVector = vec)

    conn = np.transpose(conn)
    conns = list(conn)



    # Saving Negative and Postive Edges Together
    conn_str = ''
    for lconn in conns:
        for index in range(len(lconn)):
            item = lconn[index]
            item = abs(item)
            if item == 0.0: item = int(item)
            if index == 0:
                conn_str = conn_str + str(item)
            else:
                conn_str = conn_str +'\t'+ str(item)
        conn_str = conn_str + '\n'

       

    # Saving Separate Relevance Edges
    conn_str_pos = ''
    conn_str_neg = ''

    for lindex in range(len(conns)):
        lconn = list(conns[lindex])

        for index in range(len(lconn)):
            item = lconn[index]

            if item == 0.0:
                item = int(item)
                if index == 0:
                    conn_str_pos = conn_str_pos + str(item)
                    conn_str_neg = conn_str_neg + str(item)
                else:
                    conn_str_pos = conn_str_pos + '\t'+ str(item)
                    conn_str_neg = conn_str_neg + '\t'+ str(item)


            else:
                if index == 0:
                    if item > 0:
                        conn_str_pos = conn_str_pos + str(item)
                        conn_str_neg = conn_str_neg + '0'
                    if item < 0:
                        conn_str_pos = conn_str_pos + '0'
                        conn_str_neg = conn_str_neg + str(-1*item)
                if index != 0:
                    if item > 0:
                        conn_str_pos = conn_str_pos +'\t'+ str(item)
                        conn_str_neg = conn_str_neg +'\t'+ '0'
                    if item < 0:
                        conn_str_pos = conn_str_pos +'\t'+ '0'
                        conn_str_neg = conn_str_neg +'\t'+ str(-1*item)

        conn_str_pos = conn_str_pos + '\n'
        conn_str_neg = conn_str_neg + '\n'

    # Adding a Check to Make sure that we save only the heatmaps for the correct classifications
    if actualNeuron ==  predNeuron:
        nameString = edgeDir+ heatmap_method +'_'+ dataset+'_e'+exampleHNum+'_y'+actualNeuron+'_yHat'+predNeuron+'_l'+clampNeuron
        if map == "abs":
            thisEdge = nameString +'.edge'
            file1 = open(thisEdge, 'w')
            file1.write(conn_str)
            file1.close()
        elif map == "pos":
            thisEdge = nameString +'_pos.edge'
            file2 = open(thisEdge, 'w')
            file2.write(conn_str_pos)
            file2.close()
        elif map == "neg":
            thisEdge = nameString +'_neg.edge'
            file3 = open(thisEdge, 'w')
            file3.write(conn_str_neg)
            file3.close()
        # The all option has bug and needs be to fixed before 'all' option is used
        elif map == "all":
            thisEdge =nameString +'.edge'
            file1 = open(thisEdge, 'w')
            file1.write(conn_str)
            file1.close()

            thisEdge = nameString +'_pos.edge'
            file2 = open(thisEdge, 'w')
            file2.write(conn_str_pos)
            file2.close()

            thisEdge = nameString +'_neg.edge'
            file3 = open(thisEdge, 'w')
            file3.write(conn_str_neg)
            file3.close()
        else:
            thisEdge = None
    return thisEdge


def avgMap(allTestMaps):
    heatmap = np.mean(allTestMaps, axis = 0)
    return heatmap

def probMap(allTestMaps):

    allTestMaps[allTestMaps < 0] = 0

    mx = np.max(allTestMaps)
    mn = np.min(allTestMaps)
    Q = (mx + mn)/2

    allTestMaps[allTestMaps > Q] = 1
    allTestMaps[allTestMaps < Q] = 0


    totalCount = allTestMaps.shape[0]
    
    out = np.sum(allTestMaps, axis = 0)

    return out/totalCount




def plotBrainNet(nodePath, edgePath, outputPath, configFile = "config.mat"):

    surfacePath = 'Surface/BrainMesh_ICBM152.nv'

    if edgePath !=  None:
        
        path = os.getcwd()

        outputFile = outputPath+'/'+edgePath[11:-5]+'.png'

        eng = matlab.engine.start_matlab(path)

        eng.BrainNet_MapCfg(surfacePath,
                            nodePath,
                            edgePath,
                            outputFile,
                            configFile)
        eng.quit()






def pass_topX_2D(arr, X, verbose=False):
    idx = bn.argpartition(arr, arr.size-X, axis=None)[-X:]
    width = arr.shape[1]
    idx = [divmod(i, width) for i in idx]
    idx.sort(key = lambda tup: tup[0])

    if verbose == True:
        print("The sorted 2D indices = ", idx)

    mat = np.zeros(arr.shape)

    for item in idx:
        mat[item[0], item[1]] = arr[item[0], item[1]]

    return mat


def part1(resPath, label = 0, disorder = "PTSD"):

    label = 'l' + str(label)

    pathList = os.listdir()
    for thisPath in pathList:
        if thisPath.find(label) != -1 and thisPath.find(disorder) != -1:
            print("...")



def permutationTest(pooled, sizeZ, sizeY, delta):
    np.random.shuffle(pooled)
    starZ = pooled[:sizeZ]
    starY = pooled[-sizeY]
    return starZ.mean() - starY.mean()

def runPermutationTest(z, y, numSamples, verbose = False):

    delta = z.mean() - y.mean()

    pooled = np.hstack([z, y])

    estimates = np.array(list(map(lambda x: permutationTest(pooled, z.size, y.size, delta),range(numSamples))))
    
    if verbose == True: print(delta, estimates) 

    diffCount = len(np.where(estimates <= delta)[0])

    hat_asl_perm = 1.0 - (float(diffCount)/float(numSamples))

    return hat_asl_perm








