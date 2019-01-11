#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ind_model import ind_model as model
import argparse
import time
from utils import makeGraphDict, makeFeatureDict, readRel, makeFeatureMatrix, makeTestFeature

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', help = 'learning rate for supervised loss', type = float, default = 0.1)
parser.add_argument('--embedding_size', help = 'embedding dimensions', type = int, default = 100)
parser.add_argument('--window_size', help = 'window size in random walk sequences', type = int, default = 3)
parser.add_argument('--path_size', help = 'length of random walk sequences', type = int, default = 10)
parser.add_argument('--batch_size', help = 'batch size for supervised loss', type = int, default = 200)
parser.add_argument('--g_batch_size', help = 'batch size for graph context loss', type = int, default = 200)
parser.add_argument('--g_sample_size', help = 'batch size for label context loss', type = int, default = 20)
parser.add_argument('--neg_samp', help = 'negative sampling rate; zero means using softmax', type = int, default = 5/6)
parser.add_argument('--g_learning_rate', help = 'learning rate for unsupervised loss', type = float, default = 1e-3)
parser.add_argument('--model_file', help = 'filename for saving models', type = str, default = 'ind.model')
parser.add_argument('--use_feature', help = 'whether use input features', type = bool, default = True)
parser.add_argument('--update_emb', help = 'whether update embedding when optimizing supervised loss', type = bool, default = True)
parser.add_argument('--layer_loss', help = 'whether incur loss on hidden layers', type = bool, default = True)

#parser.add_argument('--graph_path', help = 'the path of graph file', type = str, default='../author_graph_dataset/link-sub_copy.txt')
#parser.add_argument('--feature_path', help='the path of feature file', type = str, default='../author_graph_dataset/first_20_node_feature.csv')

parser.add_argument('--graph_path', help = 'the path of graph file', type = str, default='../author_graph_dataset/author-1900-2020-link-all_copy.txt')
parser.add_argument('--feature_path', help='the path of feature file', type = str, default='../author_graph_dataset/node-feature.csv')

parser.add_argument('--rel_train_path', help='the path of training relation file', type = str, default='../author_graph_dataset/5-folder-rel/rel-train1.txt')
parser.add_argument('--rel_test_path', help='the path of testing relation file', type = str, default='../author_graph_dataset/5-folder-rel/rel-test1.txt')
parser.add_argument('--embedding_path', help='the save path of embedding file', type = str, default='../author_graph_dataset/planetoid_embedding1-1.txt')


args = parser.parse_args()

def comp_accu(tpy, ty):
    import numpy as np
    return (np.argmax(tpy, axis = 1) == np.argmax(ty, axis = 1)).sum() * 1.0 / tpy.shape[0]


def main():
    start_time = time.time()
    #x, y, tx, ty, allx, graph = tuple(OBJECTS)
    graph, maxindex = makeGraphDict(args.graph_path)
    features = makeFeatureDict(args.feature_path)
    x, y = readRel(args.rel_train_path)
    tx, ty = readRel(args.rel_test_path)
    print("make feature matrix according to graph node")
    allx = makeFeatureMatrix(features, graph)
    print(allx.shape)
    tx1, tx2 = makeTestFeature(tx, features)


    m = model(args)                                                 # initialize the model
    m.add_data(x, y, allx, graph, features, maxindex)                                   # add data
    m.build()                                                       # build the model
    #m.init_train(init_iter_label = 10000, init_iter_graph = 400)    # pre-training
    m.init_train(init_iter_label=1, init_iter_graph=1)  # pre-training
    iter_cnt, max_accu = 0, 0
    while iter_cnt < 16:
        m.step_train(max_iter = 10, iter_graph = 0.1, iter_inst = 1, iter_label = 0) # perform a training step
        tpy = m.predict(tx1, tx2)                                                         # predict the dev set

        accu = comp_accu(tpy, ty)                                                   # compute the accuracy on the dev set
        print (iter_cnt, accu, max_accu)
        iter_cnt += 1
        if accu > max_accu:
            max_accu = max(max_accu, accu)


    m.extract_embedding(embeddingpath=args.embedding_path)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()


