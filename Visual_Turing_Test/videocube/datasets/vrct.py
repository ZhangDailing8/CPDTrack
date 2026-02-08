from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import six
import json
import pandas as pd

datasets_attribute = {
    'lasot': '/mnt/second/sot/lasot/attribute',
    'videocube': '/mnt/second/sot/videocube/attribute',
    'votlt2019': '/mnt/second/votlt2019/VOTLT2019/attribute',
    'got10k': '/mnt/second/sot/got10k/attribute'
}


class VRCT(object):

    def __init__(self, subset, version):
        super(VRCT, self).__init__()

        self.subset = subset

        self.version = version # set the version as 'tiny' or 'full'

        if self.version == 'tiny':
            seqs_file = os.path.join(os.path.split(os.path.realpath(__file__))[0],'ltt_git_tiny.json')
        else:
            seqs_file = os.path.join(os.path.split(os.path.realpath(__file__))[0],'ltt_git.json')


        self.infos = pd.read_csv(seqs_file)

        self.seq_names = self.infos['filenames'].tolist()
        self.sub_datasets = self.infos['datasets'].tolist()
        
        for i in range(len(self.sub_datasets)):
            if self.sub_datasets[i] == 'videocube':
                self.seq_names[i] = str(self.seq_names[i]).zfill(3)
                self.infos['filenames'][i] = str(self.seq_names[i]).zfill(3)
                
        self.anno_files = [os.path.join(datasets_attribute[sub_dataset], 'groundtruth','{}.txt'.format(s)) for s, sub_dataset in zip(self.seq_names, self.sub_datasets)]
        self.restart_files = [os.path.join(datasets_attribute[sub_dataset], 'restart','{}.txt'.format(s)) for s, sub_dataset in zip(self.seq_names, self.sub_datasets)]

        self.absent_files = [os.path.join(datasets_attribute[sub_dataset], 'absent','{}.txt'.format(s)) for s, sub_dataset in zip(self.seq_names, self.sub_datasets)]
        self.shotcut_files = [os.path.join(datasets_attribute[sub_dataset], 'shotcut','{}.txt'.format(s)) for s, sub_dataset in zip(self.seq_names, self.sub_datasets)]

    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple:
                (img_files, anno, restart_flag), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array, while
                ``restart_flag`` is a list of
                restart frames.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        seq_name = self.seq_names[index]

        img_files = self.get_frames(seq_name)

        anno = np.loadtxt(self.anno_files[index], delimiter=',')

        restart_flag = np.loadtxt(self.restart_files[index], delimiter=',', dtype=int)

        return img_files, anno, restart_flag
        
    def __len__(self):
        return len(self.seq_names)
    
    def get_frames(self, seq_name):

        sub_dataset = self.get_subdataset(seq_name).iloc[0]

        if sub_dataset == 'lasot':
            class_name = seq_name.split('-')[0]
            frame_dir = f'/mnt/second/lasot/LaSOTBenchmark/{class_name}/{seq_name}/img/'   
        elif sub_dataset == 'videocube':
            frame_dir = f'/mnt/first/hushiyu/SOT/VideoCube/val/{seq_name}/frame_{seq_name}/'
        elif sub_dataset == 'votlt2019':
            frame_dir = f'/mnt/second/votlt2019/VOTLT2019/data/{seq_name}/color/'
        elif sub_dataset == 'got10k':
            frame_dir = f'/mnt/second/got10k/val/{seq_name}/'
        else:
            raise ValueError('Unknown dataset {}'.format(sub_dataset))
        
        img_files = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))

        return img_files


    def get_subdataset(self, seq_name):
        row = self.infos[self.infos['filenames'] == seq_name]
        sub_dataset = row['datasets']

        return sub_dataset

    def get_absent(self, seq_name):
        if isinstance(seq_name, six.string_types):
            if not seq_name in self.seq_names:
                raise Exception('Sequence {} not found.'.format(seq_name))
            index = self.seq_names.index(seq_name)

        absent = np.loadtxt(self.absent_files[index], delimiter=',')

        absent_pd = pd.DataFrame(absent, columns=['absent'])

        return absent_pd

    def get_shotcut(self, seq_name):
        if isinstance(seq_name, six.string_types):
            if not seq_name in self.seq_names:
                raise Exception('Sequence {} not found.'.format(seq_name))
            index = self.seq_names.index(seq_name)

        shotcut_file = self.shotcut_files[index]

        absent = self.get_absent(seq_name)

        if not os.path.exists(shotcut_file):
            shotcut = np.zeros_like(absent)
        else:
            shotcut = np.loadtxt(shotcut_file, delimiter=',')

        shotcut_pd = pd.DataFrame(shotcut, columns=['shotcut'])

        return shotcut_pd
    
    def get_attribute(self, seq_name, attribute_name):
        if isinstance(seq_name, six.string_types):
            if not seq_name in self.seq_names:
                raise Exception('Sequence {} not found.'.format(seq_name))
            index = self.seq_names.index(seq_name)

        sub_dataset = self.get_subdataset(seq_name).iloc[0]

        attribute_file = os.path.join(datasets_attribute[sub_dataset], attribute_name, '{}.txt'.format(seq_name))
        attribute = np.loadtxt(attribute_file, delimiter=',')
        # 其中inf转换为max
        attribute[attribute == np.inf] = np.max(attribute[attribute != np.inf])

        # 转换为pd
        attribute = pd.DataFrame(attribute, columns=[attribute_name])

        return attribute
    

    def get_git_metric(self, seq_name):
        row = self.infos[self.infos['filenames'] == seq_name]
        git_metric = row['git_metric'].iloc[0]

        return git_metric
    

    def get_git_metric_num(self, num):
        git_metric = self.infos['git_metric']
        git_metric_quantile = git_metric.quantile(num)

        return git_metric_quantile


    def get_git_metric_thr(self, nbins, mode='raw'):
        
        if mode == 'raw':
            git_metric_min = self.infos['git_metric'].min()
            git_metric_max = self.infos['git_metric'].max()
            git_metric_thr = np.linspace(git_metric_min, git_metric_max, nbins)[np.newaxis, :]
        elif mode == 'quantile':
            
            inter = 1/nbins
            
            git_metric_thr = [self.get_git_metric_num(i) for i in np.arange(0,1,inter)]

            git_metric_thr = np.array(git_metric_thr)[np.newaxis, :]

        return git_metric_thr

