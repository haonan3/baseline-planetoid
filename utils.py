import random
import sys

import numpy as np
import collections
from tqdm import tqdm

#x, y, tx, ty, allx, graph

def makeGraphDict(graphpath):
    nlines = 0
    graph = collections.defaultdict(list)

    with open(graphpath, "r") as graphfile:
        for l in graphfile:
            nlines += 1

    print("Reading graph file...")
    maxindex = 0
    with open(graphpath, "r") as graphfile:
        for l in tqdm(graphfile, total=nlines):
            line = [int(i) for i in l.replace("\n", "").split(",")]
            node1, node2 = line[0], line[1]
            # undirected graph
            graph[node1].append(node2)
            graph[node2].append(node1)
            if node1 > maxindex:
                maxindex = node1
            elif node2 > maxindex:
                maxindex = node2
    return graph, maxindex


def makeFeatureDict(featurepath):
    nlines = 0
    features = collections.defaultdict(list)
    with open(featurepath, "r") as featurefile:
        for l in featurefile:
            nlines += 1
    print("Reading features file...")
    with open(featurepath, "r") as featurefile:
        for l in tqdm(featurefile, total=nlines):
            line = [float(i) for i in l.replace("\n", "").replace(","," ").split(" ")]
            node, feature = line[0], line[1:]
            features[int(node)] = np.array(feature).reshape(1,300)
    return features


def readRel(relpath, label_ratio):
    nlines = 0
    label_ratio = int(label_ratio)/10
    print("label ratio: {}".format(label_ratio))
    with open(relpath, "r") as relfile:
        for l in relfile:
            nlines += 1
    edge = []
    y = []
    with open(relpath, "r") as relfile:
        for l in tqdm(relfile, total=nlines):
            line = [int(i) for i in l.replace("\n", "").split("\t")]
            node1, node2, label = line[0], line[1], line[2]
            edge.append([node1, node2])
            label2y = np.zeros((1, 2))
            label2y[0,label] = 1
            y.append(label2y)

    start = 0
    end = int(len(edge) * label_ratio)
    return np.array(edge)[start:end,:], np.array(y).reshape((-1,2))[start:end,:]


def makeFeatureMatrix(features, graph):
    feature_matrix = []
    for key in graph:
        if key not in features:
            print("there are node not in feature file")
            features[key] = np.random.rand(1,300)
            #sys.exit()
        feature_matrix.append(features[key])
    return np.array(feature_matrix).reshape(-1,300)


def makeTestFeature(tx, features):
    TestFeature1 = []
    TestFeature2 = []
    for i in range(tx.shape[0]):
        if tx[i,0] not in features:
            print("there are test node not in feature file, makeTestFeature")
            print(tx[i,0])
            #sys.exit()
        TestFeature1.append(features[tx[i,0]])
        if tx[i,1] not in features:
            print("there are test node not in feature file, makeTestFeature")
            print(tx[i,1])
            #sys.exit()
        TestFeature2.append(features[tx[i,1]])
    return np.array(TestFeature1).reshape((-1,300)), np.array(TestFeature2).reshape((-1,300))