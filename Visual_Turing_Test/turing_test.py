import os
import json
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt


def get_gt_framelist(dataset, seq, start, end, dataset_path):
    if dataset == 'LaSOT' or dataset == 'lasot':
            class_name = seq.split('-')[0]
            frames_path = os.path.join(dataset_path, 'data', class_name, seq, 'img')
            gt_path = os.path.join(dataset_path, 'attribute', 'groundtruth', '{}.txt'.format(seq))

            frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(start+1, end+1)]
            gt = np.loadtxt(gt_path, delimiter=',')[start:end, :]

    elif dataset == 'VOTLT2019' or dataset == 'votlt2019':
        frames_path = os.path.join(dataset_path, 'data', seq, 'color')
        gt_path = os.path.join(dataset_path, 'attribute', 'groundtruth', '{}.txt'.format(seq))

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(start+1, end+1)]
        gt = np.loadtxt(gt_path, delimiter=',')[start:end, :]   

    elif dataset == 'VideoCube' or dataset == 'videocube':
        seq = str(seq).zfill(3)
        with open('./ltt_git/videocube.json', 'r', encoding='utf-8') as f:
            infos = json.load(f)['full']
        for split in ['train', 'val', 'test']:
            if seq in infos[split]:
                split_name = split
                break
        frames_path = os.path.join(dataset_path, 'data', split_name, seq, 'frame_{}'.format(seq))
        gt_path = os.path.join(dataset_path, 'attribute', 'groundtruth', '{}.txt'.format(seq))

        frames_list = ['{}/{:06d}.jpg'.format(frames_path, frame_number) for frame_number in range(start, end)]
        gt = np.loadtxt(gt_path, delimiter=',')[start:end, :]
    
    elif dataset == 'got10k':
        frames_path = os.path.join(dataset_path, 'data', 'val', seq)
        gt_path = os.path.join(dataset_path, 'attribute', 'groundtruth', '{}.txt'.format(seq))

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(start+1, end+1)]
        gt = np.loadtxt(gt_path, delimiter=',')[start:end, :]

    else:
        print('dataset:', dataset)
        raise ValueError
    
    return gt, frames_list


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # print(f"Mouse Coordinates: X: {x} Y: {y}")
        param['coordinates'][param['image_now']].append(np.array([x, y]))
    # 如果鼠标左键点击，则暂停直到左键再次点击
    elif event == cv2.EVENT_LBUTTONDOWN:
        # 当鼠标左键点击时，切换暂停状态
        param['pause_recording'] = not param.get('pause_recording', False)



def main_whole():

    exp = 'Exp'

    dataset_root = ''

    settings = {'spatial_mode': 'whole', 
                'temporal_mode': 'tem',
                'group': 'test',
                'search_size': 256,
                'template_size': 256,
                'search_factor': 4.0,
                'template_factor': 4.0,
                'fps': 15}
    
    

    res_dir = './turing_test/{}_{}_{}'.format(settings['temporal_mode'], settings['spatial_mode'], settings['group'])

    # 查找./turing_test/下目录，并确定当前是第几次测试
    dirs = os.listdir(res_dir)
    dirs = [int(dir) for dir in dirs if dir.isdigit()]
    if dirs == []:
        test_time = 1
    else:
        test_time = max(dirs) + 1

    # 创建目录
    res_path = os.path.join(res_dir, exp)

    if os.path.exists(res_path):
        pass
    else:
        os.makedirs(res_path)


    seq_info = pd.read_csv('./ltt_git/ltt_git_tiny.csv')

    test_info = pd.read_csv('./ltt_git/ltt_git_tiny_test.csv')

    # 随机打乱
    seq_info = seq_info.sample(frac=1)

    seq_info = pd.concat([test_info, seq_info])

    # 保存
    seq_info.to_csv(f'./ltt_git/tiny/{exp}.csv', index=False)


    coordinates = dict()
    states = dict()

    # turing_mode = 'origin' # pytracking, fctracker, origin
    for index, row in seq_info.iterrows():
        dataset = row['datasets']
        seq = row['filenames']
        start = 0
        end = int(row['lengths'])

        fps = settings['fps']
        if dataset == 'got10k':
            fps = settings['fps'] // 3

        dataset_path = os.path.join(dataset_root, dataset.lower())

        if dataset == 'VideoCube' or dataset == 'videocube':
            seq = str(seq).zfill(3)

        gt, frames_list = get_gt_framelist(dataset, seq, start, end, dataset_path)

        mouse_param = {'coordinates': dict(), 'image_now': '000000', 'pause_recording': False}

        state = gt[0, :]

        states[seq] = np.zeros((len(gt), 4))

        mouse_param['coordinates'][0] = []


        img_init_path = frames_list[0]
        img_init = cv2.imread(img_init_path)
        image_h, image_w, _ = img_init.shape
        print(img_init_path)
        print(gt[0, :])

        cv2.namedWindow(f'{seq} Window')
        cv2.setMouseCallback(f'{seq} Window', mouse_callback, mouse_param)


        for i in range(0, len(frames_list)):
            # print(frames_list[i])
            img = cv2.imread(frames_list[i])

            H, W, _ = img.shape
                
            if settings['spatial_mode'] == 'whole':
                # pass
                resize_factor = 1.0
                if i == 0:
                    center = gt[0, :2] + gt[0, 2:]/2
                    mouse_param['coordinates'][0].append(center.astype(np.int32))
                    print('mouse_param', mouse_param['coordinates'])
                    cv2.rectangle(img, (int(gt[0, 0]), int(gt[0, 1])), (int(gt[0, 0]+gt[0, 2]), int(gt[0, 1]+gt[0, 3])), (0, 255, 0), 2)
                else:
                    mouse_param['coordinates'][i] = []
                    


            cv2.putText(img, f'{str(i).zfill(6)}/{str(len(frames_list)).zfill(6)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            mouse_param['image_now'] = i

            cv2.imshow(f'{seq} Window', img)

            if i == 0:
                # 暂停直到按下空格键
                key = cv2.waitKey(0) & 0xFF

            while True:

                key = cv2.waitKey(1000//fps) & 0xFF
                if not mouse_param.get('pause_recording', False):
                    break

            if mouse_param['coordinates'][i] == [] and i != 0:
                mouse_param['coordinates'][i].append(mouse_param['coordinates'][i-1][-1])

            # print('mouse_param', mouse_param['coordinates'])

            if np.any(gt[i, 2:] == 0):
                pred_box = np.concatenate((mouse_param['coordinates'][i][-1], state[2:] * resize_factor))
            else:
                pred_box = np.concatenate((mouse_param['coordinates'][i][-1], gt[i, 2:] * resize_factor))


            if settings['spatial_mode'] == 'whole':
                state = pred_box
                state[0] = pred_box[0] - pred_box[2] / 2
                state[1] = pred_box[1] - pred_box[3] / 2

            states[seq][i, :] = state

            if key == ord('q'):  # 按 'q' 键退出
                break

        cv2.destroyAllWindows()

        coordinates[seq] = mouse_param['coordinates']

        print(coordinates[seq])
        print(states[seq], states[seq].shape)

        for frame in coordinates[seq].keys():
            for i in range(len(coordinates[seq][frame])):
                coordinates[seq][frame][i] = coordinates[seq][frame][i].tolist()
        
        print('seq states shape', states[seq].shape)

        # 保存坐标和状态
        np.savetxt(os.path.join(res_path, '{}.txt'.format(seq)), states[seq], delimiter=',')

        f = open(os.path.join(res_path, 'coordinates_{}.json').format(seq), 'w', encoding='utf-8')
        json.dump(coordinates, f)
        f.close()

        f = open(os.path.join(res_path, 'settings_{}.json').format(seq), 'w', encoding='utf-8')
        json.dump(settings, f)
        f.close()


    f = open(os.path.join(res_path, 'coordinates.json'), 'w', encoding='utf-8')
    json.dump(coordinates, f)
    f.close()


   




if __name__ == '__main__':

    main_whole()

