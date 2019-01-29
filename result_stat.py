import argparse
import numpy as np

def analyze_log(log_path, save_path):
    with open(log_path, "r") as logfile:
        train = []
        test = []
        for l in logfile:
            line = [float(i) for i in l.replace("\n","").split(" ")]
            train_acc, test_acc = line[0], line[1]
            train.append(train_acc)
            test.append(test_acc)
        train = np.array(train)
        test = np.array(test)
        train_std = np.std(train)
        test_std = np.std(test)
        avg_train_acc = np.mean(train)
        avg_test_acc = np.mean(test)

    with open(save_path, "w") as savefile:
        savefile.write("avg_train_acc \t avg_test_acc \t train_std \t test_std\n")
        savefile.write(str(avg_train_acc) + "\t" + str(avg_test_acc) + "\t" + str(train_std) + "\t" + str(test_std))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--planetoid_log_path', type=str, default="../author_graph_dataset/planetoid_log_path.txt", help='planetoid log file path')
    parser.add_argument('--stat_save_path', type=str, default="../author_graph_dataset/planetoid_eval_stat.txt", help='the statistical result of planetoid log')


