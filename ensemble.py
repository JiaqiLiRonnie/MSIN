import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='ntu/xsub', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')

arg = parser.parse_args()

dataset = arg.datasets
label = open('./data/val_label_tiny.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open("./weights/ntu120_setup_tiny/work_dir_joint/ntu120/epoch1_test_score.pkl", 'rb')
r1 = list(pickle.load(r1).items())
r2 = open("./weights/ntu120_setup_tiny/work_dir_bone/ntu120/epoch1_test_score.pkl", 'rb')
r2 = list(pickle.load(r2).items())
r3 = open("./weights/ntu120_setup_tiny/work_dir_object/ntu120/epoch1_test_score.pkl", 'rb')
r3 = list(pickle.load(r3).items())
r4 = open("./weights/ntu120_setup_tiny/work_dir_pairwise/ntu120/epoch1_test_score.pkl", 'rb')
r4 = list(pickle.load(r4).items())

right_num = total_num = right_num_5 = 0

for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    _, r33 = r3[i]
    _, r44 = r4[i]
    r = r11 + r22 + r33+ r44
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
	
acc = right_num / total_num
acc5 = right_num_5 / total_num

print(acc, acc5)
