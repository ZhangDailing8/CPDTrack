import os
import numpy as np
import json
import pandas as pd
import copy
from dataset_utils import _get_lasot_sequence_list as get_lasot_test
from dataset_utils import get_sequence_list
from dataset_utils import datasets_path





def ltt_git_find(dataset):

    seq_list, _ = get_sequence_list(dataset)
    print(seq_list)

    ltt_git_seqs = pd.DataFrame(columns = ['datasets', 'filenames', 'ltt_metric', 'git_metric'])

    # print(datasets_path)

    for seq in seq_list:
        absent_file = os.path.join(datasets_path[dataset], 'attribute', 'absent', '{}.txt'.format(seq))
        absent = np.loadtxt(absent_file, dtype=np.int32)
        absent_diff = np.diff(absent)
        absent_diff = np.insert(absent_diff, 0, 0)
        
        ltt_metric = np.sum(absent_diff == 1) * np.sum(absent == 1) / len(absent)

        shotcut_file = os.path.join(datasets_path[dataset], 'attribute', 'shotcut', '{}.txt'.format(seq))
        if os.path.exists(shotcut_file):
            shotcut = np.loadtxt(shotcut_file, dtype=np.int32)
            shotcut_diff = np.diff(shotcut)
            shotcut_diff = np.insert(shotcut_diff, 0, 0)
        else:
            shotcut = np.zeros_like(absent)
            shotcut_diff = np.zeros_like(absent_diff)

        git_metric = (np.sum(shotcut_diff == 1) + np.sum(absent_diff == 1)) * np.sum(absent == 1) / (len(shotcut) * len(shotcut))

        length = len(absent)
        
        ltt_git_seqs = pd.concat([ltt_git_seqs, pd.DataFrame([[dataset, seq, ltt_metric, git_metric, length]], columns=['datasets', 'filenames', 'ltt_metric', 'git_metric', 'lengths'])], ignore_index=True)

    ltt_git_seqs.to_csv('./ltt_git/{}.csv'.format(dataset), mode='w', header=True, index=False)



def ltt_git_concat():
    datasets = ['lasot', 'votlt2019', 'videocube']

    ltt_git_seqs = pd.DataFrame(columns = ['datasets', 'filenames', 'ltt_metric', 'git_metric'])

    for dataset in datasets:
        seq_info = pd.read_csv('./ltt_git/{}.csv'.format(dataset))

        # 删去其中ltt_metric和git_metric都为0的行
        seq_info = seq_info[(seq_info['ltt_metric'] != 0) | (seq_info['git_metric'] != 0)]

        ltt_git_seqs = pd.concat([ltt_git_seqs, seq_info], ignore_index=True)

    ltt_git_seqs.to_csv('./ltt_git/concat.csv', mode='w', header=True, index=False)








def sample_dataset_stage(seq_info):

    seq_info['stage'] = pd.qcut(seq_info['git_metric'], q=4, labels=False)

    seq_info['stage'] = seq_info['stage'].replace(1, 0)
    
    seq_info_sample = pd.DataFrame()
    for i in range(4):
        if i == 1:
            continue
        seq_info_stage = seq_info[seq_info['stage'] == i]

        seq_info_stage_sample = seq_info_stage.sample(n=5, weights='git_metric', random_state=0)
        seq_info_sample = pd.concat([seq_info_sample, seq_info_stage_sample])
    
    return seq_info_sample




def analysis_dataset():

    
    # dataset
    ltt_git_file_path = './ltt_git/ltt_git.csv'

    seqs_info = pd.read_csv(ltt_git_file_path)

    # 根据git_metric排序
    seqs_info = seqs_info.sort_values(by='git_metric', ascending=False)

    seqs_tiny_info = sample_dataset_stage(seqs_info)

    print(seqs_tiny_info)

    seqs_tiny_info.to_csv('./ltt_git/ltt_git_tiny.csv', index=False)




if __name__ == "__main__":

    ltt_git_find('lasot')
    ltt_git_find('votlt2019')
    ltt_git_find('videocube')

    ltt_git_concat()

    analysis_dataset

