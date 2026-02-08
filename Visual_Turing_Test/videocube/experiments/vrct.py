from __future__ import absolute_import, division, print_function

import os
import numpy as np

import time
import shutil

import json
import matplotlib.pyplot as plt
import matplotlib

from ..datasets import VRCT
from ..utils.metrics import center_error,normalized_center_error, iou, diou, giou, gts_consistency, auc_gts_consistency
from ..utils.help import makedir
from ..utils.ioutils import compress
import cv2 as cv
import pandas as pd

class ExperimentVRCT(object):
    def __init__(self, save_dir, version, trackers_pred_path, eva='in'):
        super(ExperimentVRCT, self).__init__()

        self.dataset = VRCT('val', version)
        self.version = version
        self.trackers_pred_path = trackers_pred_path

        self.result_dir = os.path.join(save_dir, 'results')
        self.time_dir = os.path.join(save_dir, 'time')
        self.img_dir = os.path.join(save_dir, 'images')

        if self.version == 'full':
            self.report_dir = os.path.join(save_dir, 'reports') 
            self.analysis_dir = os.path.join(save_dir, 'analysis')
        elif self.version == 'tiny':
            self.report_dir = os.path.join(save_dir, 'reports-tiny') 
            self.analysis_dir = os.path.join(save_dir, 'analysis-tiny')

        self.nbins_iou = 101 # set 101 points in drawing success plot
        self.nbins_ce = 401 # set 401 points in drawing original precision plot (the 401 is the top threshold value in calculating the PRE)
        self.ce_threshold = 20 # original precision plot selects 20 pixels as threshold

        self.nbins_git_metric = 101

        # makedir(save_dir)
        # makedir(self.result_dir)
        makedir(self.report_dir)
        # makedir(self.time_dir)
        makedir(self.analysis_dir)
        makedir(self.img_dir)

        self.eva = eva
        
    def report(self, tracker_names, attribute_name):

        assert isinstance(tracker_names, (list, tuple))

        analysis_dir = self.analysis_dir

        report_dir = self.report_dir

        performance = {}
        for name in tracker_names:
            # performance[name] = []

            single_report_file = os.path.join(analysis_dir, f'{name}_{attribute_name}.json')

            if os.path.exists(single_report_file):
                f = open(single_report_file, 'r', encoding='utf-8')
                single_performance = json.load(f)
                performance.update({name:single_performance})
                f.close()
                print('Existing result in {}'.format(name))
                continue
            else:
                performance.update({name: {
                    'overall': {},
                    'seq_wise': {}}})
                
            seq_num = len(self.dataset)

            # save the ious, dious and gious for success plot
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            succ_dcurve = np.zeros((seq_num, self.nbins_iou))
            succ_gcurve = np.zeros((seq_num, self.nbins_iou))

            # save the original precision value for original precision plot
            prec_curve = np.zeros((seq_num, self.nbins_ce))
            # save the novel precision value for normalized precision plot
            norm_prec_curve = np.zeros((seq_num, self.nbins_ce))

            # save average speed for each video
            speeds = np.zeros(seq_num)

            # save the normalize precision score
            norm_prec_score  = np.zeros(seq_num)

            for s in range(len(self.dataset.seq_names)):
                seq_name = self.dataset.seq_names[s]

                # print('vrct report seq_name', seq_name)
                # get the information of selected video
                img_files, anno, _ = self.dataset[seq_name]

                print('Evaluate tracker {} in video num {} with {} attribute '.format(name, seq_name, attribute_name))

                absent = self.dataset.get_absent(seq_name)
                shotcut = self.dataset.get_shotcut(seq_name)

                if attribute_name != 'normal':
                    attribute = self.dataset.get_attribute(seq_name, attribute_name)

                img_height = cv.imread(img_files[0]).shape[0]
                img_width = cv.imread(img_files[0]).shape[1]
                img_resolution = (img_width,img_height)
                bound = img_resolution

                sub_dataset = self.dataset.get_subdataset(seq_name).iloc[0]
                if sub_dataset == 'videocube':
                    sub_dataset = 'videocube_val'


                boxes = np.loadtxt(os.path.join(self.trackers_pred_path[name], sub_dataset, f'{seq_name}.txt'))


                anno = np.array(anno)
                boxes = np.array(boxes)

                for box in boxes:
                    # correction of out-of-range coordinates
                    box[0] = box[0] if box[0] > 0 else 0
                    box[2] = box[2] if box[2] < img_width - box[0] else img_width - box[0]
                    box[1] = box[1] if box[1] > 0 else 0
                    box[3] = box[3] if box[3] < img_height - box[1] else img_height - box[1]

                assert boxes.shape == anno.shape

                # calculate ious, gious, dious for success plot
                # calculate center errors and normalized center errors for precision plot
                seq_ious, seq_dious, seq_gious, seq_center_errors, seq_norm_center_errors, flags = self._calc_metrics(boxes, anno, bound)


                seq_ious = pd.DataFrame(seq_ious, columns = ['seq_ious'])
                seq_dious = pd.DataFrame(seq_dious, columns = ['seq_dious'])
                seq_gious = pd.DataFrame(seq_gious, columns = ['seq_gious'])
                seq_center_errors = pd.DataFrame(seq_center_errors, columns = ['seq_center_errors'])
                seq_norm_center_errors = pd.DataFrame(seq_norm_center_errors, columns= ['seq_norm_center_errors'])
                flags = pd.DataFrame(flags, columns = ['flags'])

                if attribute_name != 'normal':
                    data = pd.concat([seq_ious,seq_dious, seq_gious,seq_center_errors,seq_norm_center_errors,flags, absent,shotcut,attribute],axis=1) 
                else:
                    data = pd.concat([seq_ious,seq_dious, seq_gious,seq_center_errors,seq_norm_center_errors,flags, absent,shotcut],axis=1) 

                # Frames without target and transition frames are not included in the evaluation
                data = data[data.apply(lambda x: x['absent']== 0 and x['shotcut']==0, axis=1)]

                if attribute_name != 'normal':
                    # Pick frames with difficult attributes
                    data = self.select_attribute_frames(data, attribute_name) 

                data = data.drop(labels=['absent','shotcut'], axis = 1)

                seq_ious = data['seq_ious']
                seq_dious = data['seq_dious']
                seq_gious = data['seq_gious']
                seq_center_errors = data['seq_center_errors']
                seq_norm_center_errors = data['seq_norm_center_errors']   
                flags = data['flags'] 

                # Calculate the proportion of all the frames that fall into area 5 (groundtruth area)
                norm_prec_score[s] = np.nansum(flags)/len(flags)

                # Save the 5 curves of the tracker on the current video
                succ_curve[s], succ_dcurve[s], succ_gcurve[s],prec_curve[s], norm_prec_curve[s] = self._calc_curves(seq_ious, seq_dious, seq_gious,seq_center_errors, seq_norm_center_errors)


                time_file = os.path.join(self.trackers_pred_path[name], sub_dataset, f'{seq_name}_time.txt')
                

                if os.path.exists(time_file):
                    times = np.loadtxt(time_file, delimiter=',')
                    times = times[~np.isnan(times)]
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.nanmean(1. / times)

                # Update the results in current video (Only save scores)
                performance[name]['seq_wise'].update({seq_name: {
                    'success_score_iou': np.nanmean(succ_curve[s]),
                    'success_score_diou': np.nanmean(succ_dcurve[s]),                    
                    'success_score_giou': np.nanmean(succ_gcurve[s]),
                    'precision_score': prec_curve[s][self.ce_threshold],
                    'norm_prec_score':norm_prec_score[s],
                    'success_rate_iou': succ_curve[s][self.nbins_iou // 2],
                    'success_rate_diou': succ_dcurve[s][self.nbins_iou // 2],
                    'success_rate_giou': succ_gcurve[s][self.nbins_iou // 2],
                    'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

            # Average each curve
            succ_curve = np.nanmean(succ_curve, axis=0)
            succ_dcurve = np.nanmean(succ_dcurve, axis=0)
            succ_gcurve = np.nanmean(succ_gcurve, axis=0)
            prec_curve = np.nanmean(prec_curve, axis=0)
            norm_prec_curve = np.nanmean(norm_prec_curve, axis=0)

            # Generate average score
            succ_score = np.nanmean(succ_curve)
            succ_dscore = np.nanmean(succ_dcurve)
            succ_gscore = np.nanmean(succ_gcurve)
            succ_rate = succ_curve[self.nbins_iou // 2]
            succ_drate = succ_dcurve[self.nbins_iou // 2]
            succ_grate = succ_gcurve[self.nbins_iou // 2]

            prec_score = prec_curve[self.ce_threshold]
            norm_prec_score = np.nansum(norm_prec_score) / np.count_nonzero(norm_prec_score)

            if np.count_nonzero(speeds) > 0:
                avg_speed = np.nansum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'success_curve_iou': succ_curve.tolist(),
                'success_curve_diou': succ_dcurve.tolist(),
                'success_curve_giou': succ_gcurve.tolist(),
                'precision_curve': prec_curve.tolist(),
                'normalized_precision_curve': norm_prec_curve.tolist(),
                'success_score_iou': succ_score,
                'success_score_diou': succ_dscore,
                'success_score_giou': succ_gscore,
                'precision_score': prec_score,
                'norm_prec_score':norm_prec_score,
                'success_rate_iou': succ_rate,
                'success_rate_diou': succ_drate,
                'success_rate_giou': succ_grate,
                'speed_fps': avg_speed})

            with open(single_report_file, 'w') as f:
                json.dump(performance[name], f, indent=4)

        # save performance
        report_file = os.path.join(report_dir, '{}_performance.json'.format(attribute_name))
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)

        self.plot_curves_([report_file], tracker_names, attribute_name, 'all')

        return performance
    

    def report_git(self, tracker_names):
        assert isinstance(tracker_names, (list, tuple))

        analysis_dir = self.analysis_dir

        report_dir = self.report_dir

        plot_path = os.path.join(report_dir, 'ltt_git')

        if os.path.exists(plot_path):
            pass
        else:
            os.makedirs(plot_path)

        performance = {}

        for name in tracker_names:

            single_report_file = os.path.join(analysis_dir, f'{name}_normal.json')

            if os.path.exists(single_report_file):
                f = open(single_report_file, 'r', encoding='utf-8')
                single_performance = json.load(f)
                performance.update({name:single_performance})
                f.close()
            else:
                raise Exception('No existing result in {}'.format(name))
            
            # single_performance
            seqs_performance = single_performance['seq_wise']


            success_score_iou = np.array([seqs_performance[seq_name]['success_score_iou'] for seq_name in self.dataset.seq_names])
            success_score_diou = np.array([seqs_performance[seq_name]['success_score_diou'] for seq_name in self.dataset.seq_names])
            success_score_giou = np.array([seqs_performance[seq_name]['success_score_giou'] for seq_name in self.dataset.seq_names])
            precision_score = np.array([seqs_performance[seq_name]['precision_score'] for seq_name in self.dataset.seq_names])
            norm_prec_score = np.array([seqs_performance[seq_name]['norm_prec_score'] for seq_name in self.dataset.seq_names])
            success_rate_iou = np.array([seqs_performance[seq_name]['success_rate_iou'] for seq_name in self.dataset.seq_names])
            success_rate_diou = np.array([seqs_performance[seq_name]['success_rate_diou'] for seq_name in self.dataset.seq_names])
            success_rate_giou = np.array([seqs_performance[seq_name]['success_rate_giou'] for seq_name in self.dataset.seq_names])

            git_metric = np.array([self.dataset.get_git_metric(seq_name) for seq_name in self.dataset.seq_names])


            # Calculate the curves of the tracker on the current video
            success_score_iou_curve, success_score_diou_curve, success_score_giou_curve, precision_score_curve, norm_prec_score_curve, success_rate_iou_curve, success_rate_diou_curve, success_rate_giou_curve, thr_git_metric =\
                self._calc_curves_git(success_score_iou, success_score_diou, success_score_giou, precision_score, norm_prec_score, success_rate_iou, success_rate_diou, success_rate_giou, git_metric)

            performance[name] = {
                'curves': {
                    'success_score_iou_curve': success_score_iou_curve.tolist(),
                    'success_score_diou_curve': success_score_diou_curve.tolist(),
                    'success_score_giou_curve': success_score_giou_curve.tolist(),
                    'precision_score_curve': precision_score_curve.tolist(),
                    'norm_prec_score_curve': norm_prec_score_curve.tolist(),
                    'success_rate_iou_curve': success_rate_iou_curve.tolist(),
                    'success_rate_diou_curve': success_rate_diou_curve.tolist(),
                    'success_rate_giou_curve': success_rate_giou_curve.tolist(),
                    'thr_git_metric': thr_git_metric.tolist(),
                },
                'overall': {
                    'success_score_iou': np.nanmean(success_score_iou),
                    'success_score_diou': np.nanmean(success_score_diou),
                    'success_score_giou': np.nanmean(success_score_giou),
                    'precision_score': np.nanmean(precision_score),
                    'norm_prec_score': np.nanmean(norm_prec_score),
                    'success_rate_iou': np.nanmean(success_rate_iou),
                    'success_rate_diou': np.nanmean(success_rate_diou),
                    'success_rate_giou': np.nanmean(success_rate_giou),
                    'speed_fps': -1
                }
            }

        git_metric_num = self.dataset.get_git_metric_num(0.6)
        git_metric_num_index = np.argmin(np.abs(thr_git_metric - git_metric_num))
        print('git_metric_num', git_metric_num)
        print('git_metric_num_index', git_metric_num_index)

        # 保存performance
        save_json = os.path.join(plot_path, 'git_performance.json')
        with open(save_json, 'w') as f:
            json.dump(performance, f, indent=4)

       
        for key in performance[tracker_names[0]]['curves'].keys():
            plt.figure(figsize=(10, 6))

            tracker_names_sort = sorted(tracker_names, key=lambda x: performance[x]['curves'][key][0][git_metric_num_index], reverse=True)


            for name in tracker_names_sort:

                # 用虚线标出git_metric_num_75处的性能
                # plt.axvline(x=git_metric_num_index, color='r', linestyle='--')

                performance_git = performance[name]['curves'][key][0][git_metric_num_index]
                label_text = f'{name} {performance_git:.3f}'

                indices = np.argsort(performance[name]['curves']['thr_git_metric'][0])

                plt.plot(indices, performance[name]['curves'][key][0], label=label_text)

                
            plt.legend()
            plt.title(key)

            plt.xlabel('percent of STDChallenge')
            y_label = 'N_PRE (%)'
            plt.ylabel(y_label)

            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, f'{key}_git.png'))
            plt.close()


    def calc_total_error_consistency(self, tracker_names1, tracker_names2, trackers_pred_path):

        machines1 = tracker_names1
        machines2 = tracker_names2

        model_num = len(machines2)

        save_dir = os.path.join(self.analysis_dir, 'consistency')

        if os.path.exists(save_dir):
            pass
        else:
            os.makedirs(save_dir)


        for machine1 in machines1:

            
            for s in range(len(self.dataset.seq_names)):
                seq_name = self.dataset.seq_names[s]

                img_files, anno, _ = self.dataset[seq_name]

                absent = self.dataset.get_absent(seq_name)
                shotcut = self.dataset.get_shotcut(seq_name)
                filter_frames = np.squeeze(np.logical_or(absent, shotcut))

                print('filter_frames', filter_frames.shape)

                sub_dataset = self.dataset.get_subdataset(seq_name).iloc[0]
                if sub_dataset == 'videocube':
                    sub_dataset = 'videocube_val'

                res1_path = os.path.join(self.trackers_pred_path[machine1], sub_dataset, f'{seq_name}.txt')
                isExist = os.path.exists(res1_path)

                if not isExist:
                    print(f'{seq_name}, No existing result in {machine1}')
                    continue

                if 'Exp' in machine1:
                    res1 = np.loadtxt(res1_path, delimiter=',')

                    res1[:, 0] = res1[:, 0] + res1[:, 2]/2
                    res1[:, 1] = res1[:, 1] + res1[:, 3]/2

                    res1 = res1[:, :2]
                else:
                    res1 = np.loadtxt(res1_path)

                anno = np.array(anno)
                res1 = np.array(res1)

                anno = anno[~filter_frames, :]
                res1 = res1[~filter_frames, :]

                seq_length = len(anno)

                
                experimenters = []
                consistency_res = np.zeros((seq_length+5,model_num))
                model_count = 0



                for machine2 in machines2:
                    
                    experimenters.append(machine2)


                    res2_path = os.path.join(self.trackers_pred_path[machine2], sub_dataset, f'{seq_name}.txt')

                    print('machine2', machine2)

                    if 'Exp' in machine2:
                        res2 = np.loadtxt(res2_path, delimiter=',')
                        print('res2', res2.shape)
                        res2[:, 0] = res2[:, 0] + res2[:, 2]/2
                        res2[:, 1] = res2[:, 1] + res2[:, 3]/2

                        res2 = res2[:, :2]


                    else:
                        res2 = np.loadtxt(res2_path)

                    
                    
                    res2 = np.array(res2)
                    res2 = res2[~filter_frames, :]
                    
                    if self.eva == 'in':
                        consistency_res[:,model_count] = gts_consistency(anno,res1,res2)
                    else:
                        consistency_res[:,model_count] = auc_gts_consistency(anno,res1,res2)

                    # print('--------------------------------------------')

                    model_count += 1

                df_res = pd.DataFrame(consistency_res, columns=experimenters)
                df_res.to_csv(os.path.join(save_dir, f'{machine1}_{seq_name}_consistency.csv'), index=False)


    def report_error_consisitency(self, tracker_names1, tracker_names2):
        res_dir = os.path.join(self.analysis_dir, 'consistency')
        save_dir = os.path.join(self.report_dir, 'consistency')

        if os.path.exists(save_dir):
            pass
        else:
            os.makedirs(save_dir)

        x_all_array = np.zeros((len(tracker_names1), len(self.dataset.seq_names), len(tracker_names2)))
        y_all_array = np.zeros((len(tracker_names1), len(self.dataset.seq_names), len(tracker_names2)))


        for trk_id1, machine1 in enumerate(tracker_names1):
            x_seq_all = [0]*15
            y_seq_all = [0]*15
            # each_seq_count = [0]*15
            seq_count = 0

            print('machine1', machine1)

            for s in range(len(self.dataset.seq_names)):
                seq_name = self.dataset.seq_names[s]

                print('seq_name', seq_name)
    
                color = ['darkorange','lightgreen','g','tan','lime','gold','olive','c','cyan','deepskyblue','steelblue','royalblue','b','m','orchid','r','peru','gray','pink']
                marker = ['.','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d']


                seq_count += 1

                consistency_path = os.path.join(res_dir, f'{machine1}_{seq_name}_consistency.csv')
                isExist = os.path.exists(consistency_path)
                if not isExist:
                    # continue
                    x_all_array[trk_id1, s, :] = np.nan
                    y_all_array[trk_id1, s, :] = np.nan
                    continue

                infos = pd.read_csv(os.path.join(res_dir, f'{machine1}_{seq_name}_consistency.csv'))

                print('infos', infos.shape)

                for i in range(infos.shape[1]):

                    x_all_array[trk_id1, s, i] = infos.iloc[-4,i]
                    y_all_array[trk_id1, s, i] = infos.iloc[-1,i]

                    # plt.scatter(x_all_array[trk_id1, s, i], y_all_array[trk_id1, s, i], s=10, c=color[i], marker=marker[i], norm=1, label=tracker_names2[i]+': '+ str(y_all_array[trk_id1, s, i])[0:4])
                    plt.scatter(x_all_array[trk_id1, s, i], y_all_array[trk_id1, s, i], s=10, c=color[i%len(color)], marker=marker[i%len(marker)], norm=1, label=tracker_names2[i]+': '+ f"{y_all_array[trk_id1, s, i]:.3f}")
                    plt.legend(loc=2, bbox_to_anchor=(1.05,1.05))

                plt.xlabel('machine accuracy(in gts)')
                plt.ylabel('Error consistency (κ)')
                
                plt.show()


                plt.tight_layout()
                plt.savefig(os.path.join(save_dir,f'kappa_{machine1}_{seq_name}.jpg'),dpi=300, bbox_inches="tight")
            
                plt.cla()
            

            x_seq_all = np.nanmean(x_all_array[trk_id1], axis=0)
            y_seq_all = np.nanmean(y_all_array[trk_id1], axis=0)

            print('x_seq_all', x_seq_all)
            print('y_seq_all', y_seq_all)


            for i in range(len(tracker_names2)):
                plt.scatter(x_seq_all[i], y_seq_all[i], s=10, c=color[i%len(color)], marker=marker[i%len(marker)], norm=1, label=tracker_names2[i]+': '+ f"{y_seq_all[i]:.3f}")
                plt.legend(loc=2, bbox_to_anchor=(1.05,1.05))


            plt.xlabel('machine accuracy(in gts)')
            plt.ylabel('Error consistency (κ)')
            
            plt.show()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir,'total_kappa_{}.jpg'.format(machine1)),dpi=300, bbox_inches="tight")
            plt.cla()


        x_all = np.nanmean(x_all_array, axis=(0,1))
        y_all = np.nanmean(y_all_array, axis=(0,1))

        print('x_all', x_all)
        print('y_all', y_all)

        # 按照y_all的值进行排序绘制
        y_all_sort_index = np.argsort(y_all)[::-1]

        for i in y_all_sort_index:
            plt.scatter(x_all[i], y_all[i], s=10, c=color[i%len(color)], marker=marker[i%len(marker)], norm=1, label=tracker_names2[i]+': '+ f"{y_all[i]:.3f}")
            plt.legend(loc=2, bbox_to_anchor=(1.05,1.05))
 
        
        print('---------------------------------------------')

        kappa = np.mean(y_all)

        x_all_min = np.min(x_all)
        x_all_max = np.max(x_all)

        plt.plot([x_all_min,x_all_max],[kappa,kappa],"r:",label = "kappa: " + str(kappa))
        plt.legend(loc=2, bbox_to_anchor=(1.05,1.05))
        plt.xlabel('machine accuracy(in gts)')
        plt.ylabel('Error consistency (κ)')

        plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,'total_machine_kappa.jpg'),dpi=300, bbox_inches="tight")

        plt.plot([0,1],[0,1],"r:")
        plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,'total_kappa_with_line.jpg'),dpi=300, bbox_inches="tight")
        plt.cla()

        plt.figure(figsize=(10, 10))

        error_consisitency = np.mean(y_all_array, axis=1)

        # 以热力图的形式绘制二维的error_consisitency
        plt.imshow(error_consisitency, cmap='Reds', interpolation='nearest')
        
        plt.colorbar()
        plt.title('error_consisitency')
        plt.xticks(range(len(tracker_names2)), tracker_names2, rotation=90)
        plt.yticks(range(len(tracker_names1)), tracker_names1)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,'error_consisitency.jpg'),dpi=300, bbox_inches="tight")



    def select_attribute_frames(self, data, attribute_name):
        """
        Pick frames with difficult attributes
        """
        # print('attribute_name', attribute_name)
        # print('data', data[attribute_name], type(data[attribute_name]), type(data[attribute_name].iloc[0]))

        # 查找data[attribute_name]其中不为float64的数据
        # print(data[attribute_name].dtype)
        if data[attribute_name].dtype != 'float64':
            # print('data_error', data[attribute_name])
            for i in range(len(data[attribute_name])):
                if type(data[attribute_name].iloc[i]) != np.float64:
                    print('data_error', data[attribute_name].iloc[i])
        # print(data[attribute_name].dtype == 'float64')
        # print(data[attribute_name].dtype == np.float64)
        # data_error = data[data[attribute_name].apply(lambda x: type(x) != np.float64)]
        # print('data_error', data_error)
        
        if attribute_name == 'ratio':
            data = data[data.apply(lambda x: x[attribute_name] <= 0.28 or x[attribute_name] >= 2.38, axis=1)]
        elif attribute_name == 'relative_scale':
            data = data[data.apply(lambda x: x[attribute_name] <=0.02 or x[attribute_name] >=0.39, axis=1)]
        elif attribute_name == 'illumination':
            data = data[data.apply(lambda x: x[attribute_name] <=0.01 or x[attribute_name] >=0.13, axis=1)] 
        elif attribute_name == 'blur_bbox':
            data = data[data.apply(lambda x: x[attribute_name] <=95, axis=1)]
        elif attribute_name == 'delta_ratio':
            data = data[data.apply(lambda x: x[attribute_name] >=0.2, axis=1)] 
        elif attribute_name == 'delta_relative_scale':
            data = data[data.apply(lambda x: x[attribute_name] >=0.01, axis=1)]
        elif attribute_name == 'delta_illumination':
            data = data[data.apply(lambda x: x[attribute_name] >=0.0012, axis=1)]
        elif attribute_name == 'delta_blur_bbox':
            data = data[data.apply(lambda x: x[attribute_name] >=250, axis=1)]
        elif attribute_name == 'fast_motion':
            data = data[data.apply(lambda x: x[attribute_name] >=0.16, axis=1)]
        elif attribute_name == 'corrcoef':
            data = data[data.apply(lambda x: x[attribute_name] <=0.75, axis=1)]


        return data

    def _calc_metrics(self, boxes, anno, bound):
        """
        Calculate the evaluation metrics.
        """
        valid = ~np.any(np.isnan(anno), axis=1)
        if len(valid) == 0:
            print('Warning: no valid annotations')
            return None, None, None
        else:
            # calculate ious, dious and gious for success plot
            ious = iou(boxes[valid, :], anno[valid, :])
            dious = diou(boxes[valid, :], anno[valid, :])
            gious = giou(boxes[valid, :], anno[valid, :])
            # calculate center error for original precision plot
            center_errors = center_error(
                boxes[valid, :], anno[valid, :])
            # calculate normalized center error for the normalized precision plot
            norm_center_errors, flags = normalized_center_error(
                boxes[valid, :], anno[valid, :], bound)

            # print('flags', flags)
            
            return ious, dious, gious, center_errors, norm_center_errors, flags
        
    def _calc_curves(self, ious, dious, gious, center_errors, norm_center_errors):
        """
        Calculate the evaluation curves.
        """
        ious = np.asarray(ious, float)[:, np.newaxis]
        dious = np.asarray(dious, float)[:, np.newaxis]
        gious = np.asarray(gious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]
        norm_center_errors = np.asarray(norm_center_errors, float)[:, np.newaxis]

        # print('-----------------------------')
        # print('_calc_curves')
        # print('ious', ious.shape)

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]
        thr_nce = np.linspace(0, 1, self.nbins_ce)[np.newaxis, :]

        # print('thr_iou', thr_iou.shape)

        bin_iou = np.greater(ious, thr_iou)
        bin_diou = np.greater(dious, thr_iou)
        bin_giou = np.greater(gious, thr_iou)
        bin_ce = np.less(center_errors, thr_ce)
        bin_nce = np.less(norm_center_errors, thr_nce)

        # print('bin_iou', bin_iou.shape)

        succ_curve = np.nanmean(bin_iou, axis=0)
        succ_dcurve = np.nanmean(bin_diou, axis=0)
        succ_gcurve = np.nanmean(bin_giou, axis=0)
        prec_curve = np.nanmean(bin_ce, axis=0)
        norm_prec_curve = np.nanmean(bin_nce, axis=0)

        # print('succ_curve', succ_curve.shape)

        return succ_curve, succ_dcurve, succ_gcurve, prec_curve, norm_prec_curve
    
    def _calc_curves_git(self, success_score_iou, success_score_diou, success_score_giou, precision_score, norm_prec_score, success_rate_iou, success_rate_diou, success_rate_giou, git_metric):
        
        attrbutes = {
            'success_score_iou': success_score_iou,
            'success_score_diou': success_score_diou,
            'success_score_giou': success_score_giou,
            'precision_score': precision_score,
            'norm_prec_score': norm_prec_score,
            'success_rate_iou': success_rate_iou,
            'success_rate_diou': success_rate_diou,
            'success_rate_giou': success_rate_giou
        }

        for key in attrbutes.keys():
            attrbutes[key] = np.asarray(attrbutes[key], float)[:, np.newaxis]

        git_metric = np.asarray(git_metric, float)[:, np.newaxis]

        # thr_git_metric = 
        git_metric_max = git_metric.max()
        git_metric_min = git_metric.min()

        # thr_git_metric = np.linspace(git_metric_min, git_metric_max, self.nbins_git_metric)[np.newaxis, :]
        thr_git_metric = self.dataset.get_git_metric_thr(self.nbins_git_metric, mode='quantile')

        # print('thr_git_metric', thr_git_metric)


        indexes = np.greater(git_metric, thr_git_metric)

        attribute_curve = {}
        for key in attrbutes.keys():
            attribute_curve[key] = np.zeros_like(thr_git_metric)


        # 不用for
        for i in range(self.nbins_git_metric):
            for key in attrbutes.keys():
                attribute_curve[key][0,i] = np.nanmean(attrbutes[key][indexes[:,i]])


                
        for key in attrbutes.keys():
            attribute_curve[key][0, -1] = attribute_curve[key][0, -2]
     
        return attribute_curve['success_score_iou'], attribute_curve['success_score_diou'], attribute_curve['success_score_giou'], attribute_curve['precision_score'], attribute_curve['norm_prec_score'], attribute_curve['success_rate_iou'], attribute_curve['success_rate_diou'], attribute_curve['success_rate_giou'], thr_git_metric


    def plot_curves_(self, report_files, tracker_names, attribute_name, rep):
        """
        Drow Plot
        """
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)
        
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))
        if rep == 'all':
            if attribute_name is not None:
                succ_file = os.path.join(report_dir, '{}_success_plot_iou_all.png'.format(attribute_name))
                succ_dfile = os.path.join(report_dir, '{}_success_plot_diou_all.png'.format(attribute_name))
                succ_gfile = os.path.join(report_dir, '{}_success_plot_giou_all.png'.format(attribute_name))
                prec_file = os.path.join(report_dir, '{}_precision_plot_all.png'.format(attribute_name))
                norm_prec_file = os.path.join(report_dir, '{}_norm_precision_plot_all.png'.format(attribute_name))
            else:
                succ_file = os.path.join(report_dir, 'overall_success_plot_iou_all.png')
                succ_dfile = os.path.join(report_dir, 'overall_success_plot_diou_all.png')
                succ_gfile = os.path.join(report_dir, 'overall_success_plot_giou_all.png')
                prec_file = os.path.join(report_dir, 'overall_precision_plot_all.png')
                norm_prec_file = os.path.join(report_dir, 'overall_norm_precision_plot_all.png')
        else:
            if attribute_name is not None:
                succ_file = os.path.join(report_dir, '{}_success_plot_iou_{}.png'.format(attribute_name,rep))
                succ_dfile = os.path.join(report_dir, '{}_success_plot_diou_{}.png'.format(attribute_name,rep))
                succ_gfile = os.path.join(report_dir,'{}_success_plot_giou_{}.png'.format(attribute_name,rep))
                prec_file = os.path.join(report_dir, '{}_precision_plot_{}.png'.format(attribute_name,rep))
                norm_prec_file = os.path.join(report_dir,'{}_norm_precision_plot_{}.png'.format(attribute_name,rep))
            else:
                succ_file = os.path.join(report_dir, 'overall_success_plot_iou_{}.png'.format(rep))
                succ_dfile = os.path.join(report_dir, 'overall_success_plot_diou_{}.png'.format(rep))
                succ_gfile = os.path.join(report_dir, 'overall_success_plot_giou_{}.png'.format(rep))
                prec_file = os.path.join(report_dir, 'overall_precision_plot_{}.png'.format(rep))
                norm_prec_file = os.path.join(report_dir, 'overall_norm_precision_plot_{}.png'.format(rep))
        
        
        key = 'overall'

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # filter performance by tracker_names
        performance = {k:v for k,v in performance.items() if k in tracker_names}

        # sort trackers by success score iou
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score_iou'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve_iou'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score_iou']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower left', bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on VideoCube (based on IoU)')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)
        
        # sort trackers by success score diou
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score_diou'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve_diou'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score_diou']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower left', bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on VideoCube (based on DIoU)')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_dfile)
        fig.savefig(succ_dfile,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

          # sort trackers by success score giou
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score_giou'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve_giou'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score_giou']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower left', bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on VideoCube (based on GIoU)')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_gfile)
        fig.savefig(succ_gfile,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

        # sort trackers by precision score
        tracker_names = list(performance.keys())
        
        prec = [t[key]['precision_score'] for t in performance.values()]

        inds = np.argsort(prec)[::-1]

        tracker_names = [tracker_names[i] for i in inds]

        # plot precision curves
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            

            legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Location error threshold',
               ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Precision plots on VideoCube')
        ax.grid(True)
        fig.tight_layout()

        print('Saving precision plots to', prec_file)
        fig.savefig(prec_file, dpi=300)

        # plot normalized precision curves
        tracker_names = list(performance.keys())
        prec = [t[key]['norm_prec_score'] for t in performance.values()]

        inds = np.argsort(prec)[::-1]

        tracker_names = [tracker_names[i] for i in inds]

        # plot normalized precision curves
        thr_nce = np.linspace(0, 1, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_nce,
                            performance[name][key]['normalized_precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['norm_prec_score']))
        matplotlib.rcParams.update({'font.size': 8.5})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Normalized location error threshold',
               ylabel='Normalized precision',
               xlim=(0, thr_nce.max()), ylim=(0, 1),
               title='Normalized precision plots on VideoCube')
        ax.grid(True)
        fig.tight_layout()

        print('Saving normalized precision plots to', norm_prec_file)
        fig.savefig(norm_prec_file, dpi=300)
    

    def _record(self, record_file, time_file, boxes, times):
        np.savetxt(record_file, boxes, fmt='%d', delimiter=',')
        print('Results recorded at', record_file)

        times = times[:, np.newaxis]
        if os.path.exists(time_file):
            exist_times = np.loadtxt(time_file, delimiter=',')
            if exist_times.ndim == 1:
                exist_times = exist_times[:, np.newaxis]
            times = np.concatenate((exist_times, times), axis=1)
        np.savetxt(time_file, times, fmt='%.8f', delimiter=',')


    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))
    


