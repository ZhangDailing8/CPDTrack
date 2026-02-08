from __future__ import absolute_import

from videocube.experiments import *

# from tracker.siamfc import TrackerSiamFC
import os

if __name__ == '__main__':

    version = 'full' # set the version as 'tiny' or 'full'

    save_dir = ''

    attributes = [
                    'normal',
                    # 'ratio',
                    # 'relative_scale',
                    # 'illumination',
                    # 'blur_bbox',
                    # 'delta_ratio',
                    # 'delta_relative_scale',
                    # 'delta_illumination',
                    # 'delta_blur_bbox',
                    # 'fast_motion',
                    # 'corrcoef',
    ]

    trackers_git = [
                    'SiamFC', 
                    'SuperDiMP',
                    'GlobalTrack', 
                    'DiMP', 
                    'ATOM', 
                    'SiamRCNN', 
                    'SPLT', 
                    'PrDiMP', 
                    'Ocean', 
                    'KeepTrack', 
                    'SiamRPN', 
                    'DaSiamRPN', 
                    'KYS'
                    ]



    trackers_pred_path = {
    'ostrack': '/mnt/second/tracking_res/ostrack/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300',
    'seqtrack': '/mnt/second/tracking_res/seqtrack/tracking_results/seqtrack/seqtrack_b256',
    'mixvit': '/mnt/second/tracking_res/mixformer/tracking_results/mixformer_vit_online/baseline',
    'stark_st': '/mnt/second/tracking_res/stark/tracking_results/stark_st/baseline',

    'cpdtrack': '/mnt/second/fctrack/test/tracking_results/cpdtrack/baseline',

    # 'Exp01': '/mnt/second/tracking_res/Exp01',
    # 'Exp02': '/mnt/second/tracking_res/Exp02',
    # 'Exp03': '/mnt/second/tracking_res/Exp03',
    # 'Exp04': '/mnt/second/tracking_res/Exp04',
    # 'Exp05': '/mnt/second/tracking_res/Exp05'
}
    
    for trk_id, tracker in enumerate(trackers_git):
        trackers_pred_path[tracker.lower()] = os.path.join('/mnt/second/tracking_res', 'git', tracker.lower(), 'tracking_results')


    tracker_names_1 = ['Exp01', 'Exp02', 'Exp03', 'Exp04', 'Exp05']
    
    tracker_names_2 = list(trackers_pred_path.keys())




    tracker_names = list(trackers_pred_path.keys())

    experiment = ExperimentVRCT(save_dir, version, trackers_pred_path, 'auc')

    for a in attributes:
        experiment.report(tracker_names, attribute_name=a)
    experiment.report_git(tracker_names)

    experiment.calc_total_error_consistency(tracker_names_1, tracker_names_2, trackers_pred_path)
    experiment.report_error_consisitency(tracker_names_1, tracker_names_2)
    