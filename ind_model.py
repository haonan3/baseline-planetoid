#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from base_model import base_model
import numpy as np
from collections import defaultdict as dd
import random
import gensim



class ind_model(base_model):
    def add_data(self, x, y, allx, graph, features, maxindex):
        
        self.x, self.y, self.allx, self.graph, self.featureDict = x, y, allx, graph, features
        self.maxindex = maxindex
        self.num_ver = self.allx.shape[1]
        #self.num_x = self.allx.shape[0]
        self.num_x = maxindex
        print(allx.shape)
        self.y_shape=y.shape[1]
        
    def build(self):
        pass
    
    def init_train(self, init_iter_label, init_iter_graph):
        """pre-training of graph embeddings.
        init_iter_label (int): # iterations for optimizing label context loss.
        init_iter_graph (int): # iterations for optimizing graph context loss.
        """
        # gx, gy, gz = next(self.label_generator)
        # x, y = next(self.inst_generator)
        self.model_l_gx = NeuralNetUnsupervised( self.num_ver,  self.num_x, self.maxindex, self.embedding_size , self.neg_samp )
        self.model_l_x = NeuralNetSupervised( self.use_feature, self.embedding_size, self.num_ver,self.y_shape,
                                                                self.layer_loss, self.model_l_gx.get_hiddenLayer())
        
        criterion = nn.CrossEntropyLoss()
        g_loss_criterion= nn.Sigmoid()
        optimizer = torch.optim.SGD(self.model_l_gx.parameters(), lr=self.learning_rate)
        
        ### gen_label_graph
        for i in range(init_iter_label):
            gx, gy, gz = next(self.label_generator)

            if self.neg_samp > 0:
                l_gy, l_gx = self.model_l_gx(torch.tensor(gx), torch.tensor(gy))
                gz=torch.tensor(gz)
                g_loss = - torch.log( g_loss_criterion( torch.sum(l_gx * l_gy, dim = 1) * gz )  ) .sum()

            else:

                l_gx = self.model_l_gx(torch.tensor(np.array(gx).float()), torch.tensor(gy))
                gy=torch.LongTensor(gy)
                g_loss = criterion(l_gx, gy)

            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()
            print ('iter label', i, g_loss)


        ### gen_graph
        for i in range(init_iter_graph):
            gx, gy, gz = next(self.graph_generator)
            if self.neg_samp > 0:
                l_gy, l_gx = self.model_l_gx(torch.tensor(np.array(gx)), torch.tensor(gy))
                gz=torch.tensor(gz)
                g_loss = - torch.log( g_loss_criterion( torch.sum(l_gx * l_gy, dim = 1) * gz )  ) .sum()
            else:
                l_gx = self.model_l_gx(torch.tensor(np.array(gx)), torch.tensor(gy))
                gy=torch.LongTensor(gy)
                g_loss = criterion(l_gx, gy)
            
            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()
            
            print ('iter graph', i, g_loss)


    def step_train(self, max_iter, iter_graph, iter_inst, iter_label):
        """a training step. Iteratively sample batches for three loss functions.
        max_iter (int): # iterations for the current training step.
        iter_graph (int): # iterations for optimizing the graph context loss.
        iter_inst (int): # iterations for optimizing the classification loss.
        iter_label (int): # iterations for optimizing the label context loss.
        """

        self.l = [self.model_l_gx, self.model_l_x]
        g_loss_criterion= nn.Sigmoid ()
        criterion = nn.CrossEntropyLoss()
        optimizer_gx=torch.optim.SGD(self.model_l_gx.parameters(), lr=self.learning_rate)
        optimizer_x=torch.optim.SGD(self.model_l_x.parameters(), lr=self.learning_rate)
        
        for _ in range(max_iter):

            ### gen_graph
            for _ in range(self.comp_iter(iter_graph)):
                gx, gy, gz = next(self.graph_generator)
                
                if self.neg_samp > 0:
                    l_gy, l_gx = self.model_l_gx(torch.tensor(np.array(gx)), torch.tensor(gy))
                    gz=torch.tensor(gz)
                    g_loss = - torch.log( g_loss_criterion( torch.sum(l_gx * l_gy, dim = 1) * gz )  ) .sum()
                else:
                    l_gx = self.model_l_gx(torch.tensor(np.array(gx)), torch.tensor(gy))
                    gy=torch.LongTensor(gy)
                    g_loss = criterion(l_gx, gy)
                    
                optimizer_gx.zero_grad()
                g_loss.backward()
                optimizer_gx.step()

            ### gen_train_inst
            for _ in range(self.comp_iter(iter_inst)):
                x1, x2, y = next(self.inst_generator)
                if self.layer_loss and self.use_feature:
                    hid_sym1, emd_sym1, hid_sym2, emd_sym2, py_sym = self.model_l_x(x1, x2)

                    loss = criterion(py_sym, torch.tensor(np.argmax(y, axis=1))).mean()
                    loss += criterion(hid_sym1, torch.tensor(np.argmax(y, axis=1))).mean()
                    loss += criterion(emd_sym1, torch.tensor(np.argmax(y, axis=1))).mean()
                    loss += criterion(hid_sym2, torch.tensor(np.argmax(y, axis=1))).mean()
                    loss += criterion(emd_sym2, torch.tensor(np.argmax(y, axis=1))).mean()

                else:
                    py_sym = self.model_l_x(x1, x2)
                    loss = criterion(py_sym, torch.tensor(np.argmax(y, axis=1))).mean()
                optimizer_x.zero_grad()
                loss.backward()
                optimizer_x.step()
                print ('iter graph', loss)

            ### gen_label_graph
            for _ in range(self.comp_iter(iter_label)):
                gx, gy, gz = next(self.label_generator)
                if self.neg_samp >  0:
                    l_gy, l_gx = self.model_l_gx(torch.tensor(np.array(gx)), torch.tensor(gy))
                    gz=torch.tensor(gz)
                    g_loss = - torch.log( g_loss_criterion( torch.sum(l_gx * l_gy, dim = 1) * gz )  ) .sum()
                    print ('iter graph', g_loss)

                else:
                    l_gx = self.model_l_gx(torch.tensor(np.array(gx)), torch.tensor(gy))
                    gy=torch.LongTensor(gy)
                    g_loss = criterion(l_gx, gy)

                optimizer_gx.zero_grad()
                g_loss.backward()
                optimizer_gx.step()


    def predict(self, tx1, tx2):
        """predict the dev or test instances.
        tx (scipy.sparse.csr_matrix): feature vectors for dev instances.
        returns (numpy.ndarray, #instacnes * #classes): classification probabilities for dev instances.
        """
        with torch.no_grad():
            if self.layer_loss and self.use_feature:
                _, _, _, _, outputs = self.model_l_x(tx1, tx2)
            else:
                outputs = self.model_l_x(tx1, tx2)
        return outputs.numpy()


    def gen_train_inst(self):
        """generator for batches for classification loss.
        """
        while True:
            ind = np.array(np.random.permutation(self.x.shape[0]), dtype = np.int32)
            i = 0
            while i < self.x.shape[0]:
                j = min(ind.shape[0], i + self.batch_size)

                # self.x is a np.array, shape:[n,2]
                # self.y is a np.array, shape:[n,]
                node1 = self.x[ind[i:j],0]
                node2 = self.x[ind[i:j],1]
                label = self.y[ind[i: j]]

                # node1's feature, node2's feature, label
                # self.allx is a np.array, shape: [n, # of feature]
                # yield self.allx[node1], self.allx[node2], label
                node1_feature = []
                node2_feature = []
                for idx in range(j-i):
                    node1_feature.append(self.featureDict[node1[idx]])
                    node2_feature.append(self.featureDict[node2[idx]])
                yield np.array(node1_feature), np.array(node2_feature), label
                i = j


    def gen_graph(self):
        """generator for batches for graph context loss.
        """
        while True:
            ind = np.random.permutation(self.num_x)
            i = 0
            while i < ind.shape[0]:
                g, gy = [], []
                j = min(ind.shape[0], i + self.g_batch_size)
                for k in ind[i:j]:
                    if len(self.graph[k]) == 0:
                        continue
                    path = [k]

                    for _ in range(self.path_size):
                        # random choice a neighbor from neighbors of the last node in the path  ---random walk
                        path.append( random.choice(self.graph[path[-1]]) )

                    for l in range(len(path)):
                        if path[l] >= self.allx.shape[0]:
                            continue
                        for m in range(l - self.window_size, l + self.window_size + 1):
                            if m < 0 or m >= len(path):
                                continue
                            if path[m] >= self.allx.shape[0]:
                                continue
                            # positive sampling
                            g.append([path[l], path[m]])
                            gy.append(1.0)
                            # negative sampling
                            for _ in range( self.comp_iter(self.neg_samp)):
                                g.append([path[l], random.randint(0, self.num_x - 1)])
                                gy.append(- 1.0)

                g = np.array(g, dtype = np.int32)
                yield self.allx[g[:, 0]], g[:, 1], gy
                i = j


    def gen_label_graph(self):
        """generator for batches for label context loss.
        modify according to our task:
        we define the node with directed link has same label
        """
        # relation type 0 is advisor-advisee
        # relation type 1 is colleague
        types = 2
        labels, label2inst, not_label = [], dd(list), dd(list)

        # for i in range(self.x.shape[0]):
        #     for j in range(types):
        #         if self.y == j:
        #             labels.append(j)
        #             label2inst[j].append(i)
        #         else:
        #             not_label[j].append(i)

        for i in range(self.x.shape[0]):
            flag = False
            for j in range(self.y.shape[1]):
                if self.y[i, j] == 1 and not flag:
                    labels.append(j)
                    label2inst[j].append(i)
                    flag = True
                elif self.y[i, j] == 0:
                    not_label[j].append(i)

        while True:
            g, gy = [], []
            for _ in range(self.g_sample_size):
                idx1 = random.randint(0, self.x.shape[0] - 1)
                label = labels[idx1]
                if len(label2inst) == 1:
                    continue

                g.append([self.x[idx1,0], self.x[idx1,1]])
                gy.append(1.0)
                for _ in range(self.comp_iter(self.neg_samp)):
                    idx2 = random.choice(not_label[label])
                    g.append([self.x[idx1,0], self.x[idx2,1]])
                    gy.append(-1.0)

            g = np.array(g, dtype = np.int32)
            features = []
            for i in g[:, 0]:
                feature = self.featureDict[i]
                features.append(feature)
            yield np.array(features).reshape((-1,300)), g[:, 1], gy


    def save_embedding(self, id, path):
        learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(self.embedding_size)
        learned_embed.add(id, self.model_l_gx.embedding_l_gy.weight.data.cpu().numpy())
        learned_embed.save_word2vec_format(fname=path, binary=False)



class NeuralNetUnsupervised(nn.Module):
    def __init__(self, num_ver, num_x, maxindex, embedding_size,neg_samp,  **kwargs):
        super(NeuralNetUnsupervised, self).__init__()
        
        self.num_ver = num_ver
        self.num_x = num_x
        self.maxindex = maxindex
        self.embedding_size = embedding_size
        self.neg_samp = neg_samp
        
        
        self.fc_gx = nn.Linear(self.num_ver, self.embedding_size)
        self.nonlinearity_gx = nn.ReLU()
        
        if self.neg_samp > 0:
            print("num_x: ",self.num_x)
            print("maxindex: ",self.maxindex)
            self.embedding_l_gy = nn.Embedding(num_embeddings = int(self.maxindex), embedding_dim = self.embedding_size)
        else :
            self.fc_gx2 = nn.Linear(self.embedding_size, self.num_x)
            self.nonlinearity_gx2 = nn.Softmax()

            
    def forward(self, gx, gy):
        l_gx = self.fc_gx(gx.float())
        l_gx = self.nonlinearity_gx(l_gx)
        
        if self.neg_samp > 0:
            l_gy = self.embedding_l_gy(gy.long())
            return l_gy, l_gx
        else:
            l_gx = self.fc_gx2(l_gx)
            l_gx = self.nonlinearity_gx2(l_gx)
            return l_gx
        
    def get_hiddenLayer(self):
        return self.fc_gx.weight
            
    
class NeuralNetSupervised(nn.Module):
    def __init__(self, use_feature, embedding_size, num_ver, y_shape, layer_loss, hiddenLayerWeight, **kwargs):
                
        super(NeuralNetSupervised, self).__init__()
        
        self.y_shape = y_shape
        self.num_ver = num_ver
        self.embedding_size = embedding_size
        self.use_feature = use_feature
        self.layer_loss = layer_loss
        self.hiddenLayerWeight = hiddenLayerWeight
        
        #node1 direction1
        self.fc_node1_x1 = nn.Linear(self.num_ver, self.y_shape )
        self.nonlinearity_node1_x1 = nn.Softmax()

        #node2 direction1
        self.fc_node2_x1 = nn.Linear(self.num_ver, self.y_shape)
        self.nonlinearity_node2_x1 = nn.Softmax()

        #node1 direction2
        self.nonlinearity_node1_x2_1 = nn.ReLU()
        self.fc_node1_x2_2 = nn.Linear(self.embedding_size, self.y_shape)
        self.nonlinearity_node1_x2_2 = nn.Softmax()

        #node2 direction2
        self.nonlinearity_node2_x2_1 = nn.ReLU()
        self.fc_node2_x2_2 = nn.Linear(self.embedding_size, self.y_shape)
        self.nonlinearity_node2_x2_2 = nn.Softmax()


        if self.use_feature:
            self.fc_cat_four_feature = nn.Linear( self.y_shape * 4 ,self.y_shape)
            self.nonlinearity_cat_four_feature = nn.Softmax()
        else:
            self.fc_cat_two_feature = nn.Linear( self.y_shape * 2 ,self.y_shape)
            self.nonlinearity_cat_two_feature = nn.Softmax()


    def forward(self, x1, x2):
            # node1 two flow direction
            x1=torch.tensor(np.array(x1))
            l_x1_1 = self.fc_node1_x1(x1)
            l_x1_1 = self.nonlinearity_node1_x1(l_x1_1)
            l_x1_2 = torch.mm(x1, self.hiddenLayerWeight.t())
            l_x1_2 = self.nonlinearity_node1_x2_1(l_x1_2)
            l_x1_2 = self.fc_node1_x2_2(l_x1_2)
            l_x1_2 = self.nonlinearity_node1_x2_2(l_x1_2)

            # node2 two flow direction
            x2 = torch.tensor(np.array(x2))
            l_x2_1 = self.fc_node2_x1(x2)
            l_x2_1 = self.nonlinearity_node2_x1(l_x2_1)
            l_x2_2 = torch.mm(x2, self.hiddenLayerWeight.t())
            l_x2_2 = self.nonlinearity_node2_x2_1(l_x2_2)
            l_x2_2 = self.fc_node2_x2_2(l_x2_2)
            l_x2_2 = self.nonlinearity_node2_x2_2(l_x2_2)


            if self.use_feature:
                l_x = torch.cat((l_x1_1, l_x1_2, l_x2_1, l_x2_2), dim=1)
                l_x = self.fc_cat_four_feature(l_x)
                l_x = self.nonlinearity_cat_four_feature(l_x)
            else:
                l_x = torch.cat((l_x1_2, l_x2_2), dim=1)
                l_x = self.fc_cat_two_feature(l_x)
                l_x = self.nonlinearity_cat_two_feature(l_x)

                
            if self.layer_loss and self.use_feature:    # layer_loss should be false
                return l_x1_1, l_x1_2, l_x2_1, l_x2_2, l_x # hid_sym1, emd_sym1, hid_sym2, emd_sym2, py_sym
            else:
                return l_x