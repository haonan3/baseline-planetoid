#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ind_model import ind_model as model
import argparse
import time
from utils import makeGraphDict, makeFeatureDict, readRel, makeFeatureMatrix, makeTestFeature

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', help = 'learning rate for supervised loss', type = float, default = 0.05)
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

parser.add_argument('--label_ratio',help='the train label ratio', default=1.0, type=float)
parser.add_argument('--rel_train_path', help='the path of training relation file', type = str, default='../author_graph_dataset/5-folder-rel/rel-train')
parser.add_argument('--rel_test_path', help='the path of testing relation file', type = str, default='../author_graph_dataset/5-folder-rel/rel-test')
parser.add_argument('--log_path', help='the path of log file', type = str, default='../author_graph_dataset/planetoid_log_path.txt')
parser.add_argument('--folder_num', help='start from 1 to 5', type = int, default=1)

#parser.add_argument('--embedding_path', help='the save path of embedding file', type = str, default='../author_graph_dataset/planetoid_embedding1-1.txt')


args = parser.parse_args()

def comp_accu(tpy, ty):
    import numpy as np
    return (np.argmax(tpy, axis = 1) == np.argmax(ty, axis = 1)).sum() * 1.0 / tpy.shape[0]

def save_log(path, max_train_accu, max_test_accu):
    with open(path, "a") as file:
        file.write(str(max_train_accu) + " " + str(max_test_accu) + "\n")

def main():
    start_time = time.time()
    #x, y, tx, ty, allx, graph = tuple(OBJECTS)
    graph, maxindex = makeGraphDict(args.graph_path)
    features = makeFeatureDict(args.feature_path)
    args.rel_train_path = args.rel_train_path+str(args.folder_num)+str('.txt')
    args.rel_test_path  = args.rel_test_path+str(args.folder_num)+str('.txt')

    x, y = readRel(args.rel_train_path, args.label_ratio)
    tx, ty = readRel(args.rel_test_path,1)
    print("make feature matrix according to graph node")
    allx = makeFeatureMatrix(features, graph)
    print(allx.shape)
    testx1, testx2 = makeTestFeature(tx, features)
    trainx1, trainx2 = makeTestFeature(x, features)



    m = model(args)                                                 # initialize the model
    m.add_data(x, y, allx, graph, features, maxindex)                                   # add data
    m.build()                                                       # build the model
    m.init_train(init_iter_label = 10000, init_iter_graph = 400)    # pre-training
    #m.init_train(init_iter_label=1, init_iter_graph=1)  # pre-training
    iter_cnt, max_test_accu, max_train_accu = 0, 0, 0
    while iter_cnt < 15000:
        m.step_train(max_iter = 10, iter_graph = 0.1, iter_inst = 1, iter_label = 0) # perform a training step
        testpy = m.predict(testx1, testx2)                                                         # predict the dev set
        trainpy = m.predict(trainx1,trainx2)
        test_accu = comp_accu(testpy, ty)                                                   # compute the accuracy on the dev set
        train_accu = comp_accu(trainpy,y)
        print("Iteration: {} | curr train acc: {}, best train acc: {} | curr test acc: {}, best test acc: {}".
              format(iter_cnt, train_accu, max_train_accu, test_accu, max_test_accu))

        iter_cnt += 1
        if test_accu >= max_test_accu:
            max_test_accu = test_accu
        if train_accu >= max_train_accu:
            max_train_accu = train_accu

    #m.extract_embedding(embeddingpath=args.embedding_path)  # save embedding
    save_log(args.log_path, max_train_accu, max_test_accu)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()


