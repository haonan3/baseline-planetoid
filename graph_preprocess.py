import argparse

import gensim
import numpy as np
from tqdm import tqdm


# Read feature for each node. e.g. feature at line 3(starting from 0) is feature of node 3.
# Return a 2-D np.array, each line is a feature for one node.
def read_feature(feature_path):
    print("Read feature file from: |" + feature_path + "\n")

    nlines = 0
    features = []
    with open(feature_path, "r") as featurefile:
        for l in featurefile:
            nlines += 1

    with open(feature_path, "r") as featurefile:
        for l in tqdm(featurefile, total=nlines):
            feature = [float(i) for i in l.replace("\n", "").split(",")]
            features.append(feature)
    return np.array(features)


# Read label for each node
def read_cluster(cluster_path):
    print("Read feature file from: |" + cluster_path + "\n")
    nlines = 0
    labels = []
    with open(cluster_path, "r") as clusterfile:
        for l in clusterfile:
            nlines += 1

    with open(cluster_path, "r") as clusterfile:
        for l in tqdm(clusterfile, total=nlines):
            line = [int(i) for i in l.replace("\n","").split(",")]
            line = np.array(line)
            label = np.argmax(line)
            labels.append(label)
    return labels


# Assume node index starts from 0.
def write_feature(features, save_path, binary=False):
    print("Save processed features to: |" + save_path + "\n")
    index_list = range(features.shape[0])
    learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(features.shape[1])
    learned_embed.add(list(index_list), features)
    learned_embed.save_word2vec_format(fname=save_path, binary=binary, total_vec=len(index_list))


# Assume node index starts from 0.
def write_cluster(labels, save_path):
    print("Save node labels to: |" + save_path + "\n")
    with open(save_path, "w") as savefile:
        for i,v in enumerate(labels):
            savefile.write(str(i) + " " + str(v) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_path', type=str, default="../../example/feature.txt", help='feature file path')
    parser.add_argument('--feature_save_path', type=str, default="../../example/preprocessed_feature.txt",
                        help='preprocessed feature file path')
    parser.add_argument('--cluster_path', type=str, default="../../example/cluster.txt", help='cluster file path')
    parser.add_argument('--cluster_save_path', type=str, default="../../example/preprocessed_cluster.txt",
                        help='preprocessed cluster file path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    features = read_feature(args.feature_path)
    write_feature(features, args.feature_save_path)
    labels = read_cluster(args.cluster_path)
    write_cluster(labels, args.cluster_save_path)