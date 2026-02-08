import torch
import math
import numpy as np
import cv2 as cv
import torch.nn.functional as F
# from lib.utils.misc import NestedTensor
from scipy.stats import norm

def calc_gt_accu(pred, gt):
    # 如果pred的中心点在gt内部，则为1，否则为0
    # pred: [:, 4] x,y,w,h
    # gt: [:, 4] x,y,w,h
    pred_center = pred[:, :2] + 0.5 * pred[:, 2:4]    
    accu = torch.zeros(pred.shape[0])

    for i in range(pred.shape[0]):
        if pred_center[i, 0] > gt[i, 0] and pred_center[i, 0] < gt[i, 0] + gt[i, 2] and \
            pred_center[i, 1] > gt[i, 1] and pred_center[i, 1] < gt[i, 1] + gt[i, 3]:
            accu[i] = 1

    return accu

def bgr2rgb(color):
    return color[..., ::-1]

def bgr2rgb_norm(color):
    # color: tuple
    return [color[2]/255.0, color[1]/255.0, color[0]/255.0]

def calc_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    # 计算交集
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
    # 计算并集
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - inter_area
    # 计算IoU
    iou = inter_area / union_area
    return iou

def sample_target(im, target_bb, search_area_factor, output_sz=None, crop_box=False):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """

    # print('output_sz', output_sz)

    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        if crop_box:
            return im_crop_padded, resize_factor, crop_sz, [x1 + x1_pad, y1 + y1_pad, x2 - x2_pad, y2 - y2_pad]
        else:
            return im_crop_padded, resize_factor, crop_sz

    else:
        if crop_box:
            return im_crop_padded, 1.0, crop_sz, [x1 + x1_pad, y1 + y1_pad, x2 - x2_pad, y2 - y2_pad]
        else:
            return im_crop_padded, 1.0, crop_sz

def adjust_pad(image_h, image_w):
    x1_pad = 0
    x2_pad = 0
    y1_pad = 0
    y2_pad = 0

    if image_w > image_h:
        y1_pad = int((image_w - image_h)/2)
        y2_pad = int(image_w - image_h - y1_pad)
    elif image_w < image_h:
        x1_pad = int((image_h - image_w)/2)
        x2_pad = int(image_h - image_w - x1_pad)
    else:
        pass

    return x1_pad, x2_pad, y1_pad, y2_pad

def gaussin_crop(gt, image_h, image_w, mode='search'):

    # print('gt', gt)

    if not isinstance(gt, list):
        x1, y1, w, h = gt.tolist()
    else:
        x1, y1, w, h = gt
    # x1 = gt[0]
    # y1 = gt[1]
    # w = gt[2]
    # h = gt[3]
        
    # out-of-plane
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x1+w > image_w:
        w = image_w - x1
    if y1+h > image_h:
        h = image_h - y1


    x_c = x1 + w/2.0
    y_c = y1 + h/2.0

    mu_x = x1 + w/2.0
    sigma_x = image_w/6.0

    cdf_value_x = norm.cdf(x1+w, mu_x, sigma_x) - norm.cdf(x1, mu_x, sigma_x)
    # cdf_value_start_x = norm.cdf(0, mu_x, sigma_x)
    # cdf_value_end_x = norm.cdf(image_w, mu_x, sigma_x)
    
    x_expand_c = x_c

    if mode == 'search':
        bbox_w_expand = w/cdf_value_x


    elif mode == 'template':
        bbox_w_expand = w/cdf_value_x*np.exp(cdf_value_x-1)
        # bbox_w_expand = w/cdf_value_x*(np.exp(cdf_value_x-1) + 0.9)/2.0
    elif mode == 'test':
        bbox_w_expand = w/cdf_value_x*(1-np.exp(-5*cdf_value_x))

    x_expand_1 = x_expand_c - bbox_w_expand/2.0
    x_expand_2 = x_expand_c + bbox_w_expand/2.0

    if x_expand_1 < 0:
        x_expand_1 = 0
    if x_expand_2 > image_w:
        x_expand_2 = image_w

    mu_y = y1 + h/2.0
    sigma_y = image_h/6.0

    cdf_value_y = norm.cdf(y1+h, mu_y, sigma_y) - norm.cdf(y1, mu_y, sigma_y)

    y_expand_c = y_c
    if mode == 'search':
        bbox_h_expand = h/cdf_value_y


    elif mode == 'template':
        bbox_h_expand = h/cdf_value_y*np.exp(cdf_value_y-1)
    elif mode == 'test':
        # bbox_h_expand = h/cdf_value_y*(np.exp(cdf_value_y-1) + cdf_value_y)/2.0
        bbox_h_expand = h/cdf_value_y*(1-np.exp(-5*cdf_value_y))

    y_expand_1 = y_expand_c - bbox_h_expand/2.0
    y_expand_2 = y_expand_c + bbox_h_expand/2.0


    if y_expand_1 < 0:
        y_expand_1 = 0
    if y_expand_2 > image_h:
        y_expand_2 = image_h

    return [int(x_expand_1), int(y_expand_1), int(x_expand_2-x_expand_1), int(y_expand_2-y_expand_1)]

def sample_target_gaussian(im, target_bb, output_sz=None, mode='search'):
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    image_h, image_w, _ = im.shape

    try:
        box_expand = gaussin_crop(target_bb, image_h, image_w, mode=mode)
    except:
        print('target_bb', target_bb)
        raise

    x_expand_1, y_expand_1, w_expand, h_expand = box_expand

    # crop_sz = math.ceil(math.sqrt(w_expand * h_expand))
    crop_sz = max(w_expand, h_expand)


    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x_expand_2 = x_expand_1 + w_expand
    y_expand_2 = y_expand_1 + h_expand

    x1_pad = 0
    x2_pad = 0
    y1_pad = 0
    y2_pad = 0

    if w_expand > h_expand:
        y1_pad = int((w_expand - h_expand)/2)
        y2_pad = int(w_expand - h_expand - y1_pad)
    elif w_expand < h_expand:
        x1_pad = int((h_expand - w_expand)/2)
        x2_pad = int(h_expand - w_expand - x1_pad)
    else:
        pass

    im_crop = im[y_expand_1:y_expand_2, x_expand_1:x_expand_2, :]

    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)

    H, W, _ = im_crop_padded.shape

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))

        # box_expand_resize = gaussin_transform_image_to_crop(torch.Tensor(box_expand), resize_factor, torch.Tensor([crop_sz]), torch.Tensor([image_h, image_w]), normalize=True)

        return im_crop_padded, resize_factor, box_expand
    else:
        return im_crop_padded, 1.0, box_expand

def sample_target_gaussian_global(im, output_sz=None, mode='search'):
    image_h, image_w, _ = im.shape

    x1_pad = 0
    x2_pad = 0
    y1_pad = 0
    y2_pad = 0

    if image_w > image_h:
        y1_pad = int((image_w - image_h)/2)
        y2_pad = int(image_w - image_h - y1_pad)
    elif image_w < image_h:
        x1_pad = int((image_h - image_w)/2)
        x2_pad = int(image_h - image_w - x1_pad)
    else:
        pass

    im_crop_padded = cv.copyMakeBorder(im, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)

    crop_sz = max(image_h, image_w)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))

        return im_crop_padded, resize_factor
    else:
        return im_crop_padded, 1.0

def sample_target_gaussian_global_show(im, target_bb, output_sz_local=None, output_sz_global=None, mode='search'):
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    image_h, image_w, _ = im.shape

    try:
        box_expand = gaussin_crop(target_bb, image_h, image_w, mode=mode)
    except:
        print('target_bb', target_bb)
        raise

    x_expand_1, y_expand_1, w_expand, h_expand = box_expand

    x_expand_2 = x_expand_1 + w_expand
    y_expand_2 = y_expand_1 + h_expand

#     # crop_sz = math.ceil(math.sqrt(w_expand * h_expand))
    crop_sz_local = max(w_expand, h_expand)
    crop_sz_global = max(image_h, image_w)

    x1_pad_local, x2_pad_local, y1_pad_local, y2_pad_local = adjust_pad(h_expand, w_expand)
    x1_pad_global, x2_pad_global, y1_pad_global, y2_pad_global = adjust_pad(image_h, image_w)

    resize_factor_local = output_sz_local / crop_sz_local
    resize_factor_global = output_sz_global / crop_sz_global

    output_sz_global_show = int(max(image_h, image_w)*resize_factor_local)

    im_crop_padded_global = cv.copyMakeBorder(im, y1_pad_global, y2_pad_global, x1_pad_global, x2_pad_global, cv.BORDER_CONSTANT)
    im_crop_padded_global = cv.resize(im_crop_padded_global, (output_sz_global, output_sz_global))
    print('output_sz_global_show', output_sz_global_show)
    im_crop_padded_global = cv.resize(im_crop_padded_global, (output_sz_global_show, output_sz_global_show))

    im_crop_padded_local = cv.copyMakeBorder(im[y_expand_1:y_expand_2, x_expand_1:x_expand_2, :], y1_pad_local, y2_pad_local, x1_pad_local, x2_pad_local, cv.BORDER_CONSTANT)
    im_crop_padded_local = cv.resize(im_crop_padded_local, (output_sz_local, output_sz_local))
    # im_crop_padded_local = cv.resize(im_crop_padded_local, (output_sz_global_show, output_sz_global_show))

    resize_factor_global_show = output_sz_global_show / crop_sz_global

    box_expand_crop_global = gaussin_transform_image_to_crop(np.array(box_expand), resize_factor_global_show, np.array([output_sz_global_show, output_sz_global_show]), np.array([image_w, image_h]), normalize=False).astype(np.int)

    print('box_expand_crop_global', box_expand_crop_global)
    print(y1_pad_local * resize_factor_global_show)
    print((y1_pad_local + h_expand) * resize_factor_global_show)
    print(x1_pad_local * resize_factor_global_show)
    print((x1_pad_local + w_expand) * resize_factor_global_show)

    im_crop_padded_global[box_expand_crop_global[1]:box_expand_crop_global[1]+box_expand_crop_global[3], box_expand_crop_global[0]:box_expand_crop_global[0]+box_expand_crop_global[2], :] = \
    im_crop_padded_local[np.ceil(y1_pad_local * resize_factor_global_show).astype(np.int):np.ceil(y1_pad_local * resize_factor_global_show).astype(np.int)+box_expand_crop_global[3], np.ceil(x1_pad_local * resize_factor_global_show).astype(np.int):np.ceil(x1_pad_local * resize_factor_global_show).astype(np.int)+box_expand_crop_global[2], :]

    return im_crop_padded_global, resize_factor_global_show, box_expand_crop_global


def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]

def map_box_back_global(pred_box: list, resize_factor: float, H, W, search_expand_size):
    # resize_factor
    cx, cy, w, h = pred_box
    cx_real = (cx - search_expand_size/2.0) / resize_factor + W/2.0
    cy_real = (cy - search_expand_size/2.0) / resize_factor + H/2.0
    w = w / resize_factor
    h = h / resize_factor
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

def map_box_back(pred_box: list, resize_factor: float, state: list, search_size: int):
    cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

def transform_image_to_crop(box_in: np.ndarray, box_extract: np.ndarray, resize_factor: float,
                            crop_sz: np.ndarray, normalize=False) -> np.ndarray:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = np.concatenate((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / (crop_sz[0]-1)
    else:
        return box_out
    
def gaussin_transform_image_to_crop(box_in: np.ndarray, resize_factor: float,
                            crop_sz: np.ndarray, origin_size: np.ndarray, normalize=False) -> np.ndarray:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    # torch.max(origin_size[0:2])
    print('origin_size', origin_size)
    print('crop_sz', crop_sz)
    print('resize_factor', resize_factor)
    print('box_in', box_in, type(box_in))

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]
    # print('origin_size', origin_size)
    # box_extract_center = torch.div(origin_size, 2, rounding_mode='floor')
    box_extract_center = origin_size // 2
    box_out_center = crop_sz / 2 + (box_in_center - box_extract_center) * resize_factor

    box_out_wh = box_in[2:4] * resize_factor

    box_out = np.concatenate((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        #crop_sz[0] - 1, modified by chenxin from crop_sz[0],2022.7.15
        return box_out / crop_sz
    else:
        return box_out


def seq_seg(arr):
    # 找到所有值为0的索引
    zero_indices = np.where(arr == 0)[0]

    # 初始化用于存储每段连续0的索引范围的列表
    index_ranges = []

    # 如果有0的话，进行处理
    if zero_indices.size > 0:
        # 计算相邻索引之间的差异
        diff = np.diff(zero_indices)
        # 差异大于1的地方即为非连续的标记
        breaks = np.where(diff > 1)[0]
        
        # 分割索引数组为连续的部分
        start_idx = 0
        for brk in breaks:
            end_idx = brk
            index_ranges.append((zero_indices[start_idx], zero_indices[end_idx]))
            start_idx = end_idx + 1
        # 添加最后一段连续的0的索引范围
        index_ranges.append((zero_indices[start_idx], zero_indices[-1]))

    return index_ranges

def attr2bin(attr, attr_array):
    if attr == 'occlusion':
        return attr_array
    elif attr == 'color_constancy_tran':
        return attr_array < 0.99
    elif attr == 'scale':
        # print('attr_array', (attr_array<50).dtype, attr_array.shape)
        # print('attr_array', (attr_array>750).dtype, attr_array.shape)
        return (attr_array < 50) | (attr_array > 750)
    elif attr == 'ratio':
        return (attr_array < 1/3) | (attr_array > 3)
    elif attr == 'delta_blur':
        return attr_array > 1.5
    elif attr == 'delta_color_constancy_tran':
        return attr_array > 0.0001
    elif attr == 'delta_scale':
        return attr_array > 30
    elif attr == 'delta_ratio':
        return attr_array > 0.2
    elif attr == 'corrcoef':
        return attr_array < 0.8
    elif attr == 'motion':
        return attr_array > 0.2


def normalized_center_error(rects1, rects2, bound):
    r"""Normalized center error.
    Novel metrics.

    Args:
        rects1 (numpy.ndarray): Prediction box. An N x 4 numpy array, each line represent a rectangle (left, top, width, height).
        rects2 (numpy.ndarray): Groudntruth box. An N x 4 numpy array, each line represent a rectangle (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    centers1 = rects1[..., :2] + (rects1[..., 2:] - 1) / 2 # prediction box
    centers2 = rects2[..., :2] + (rects2[..., 2:] - 1) / 2 # groundtruth box
    width, height = bound

    # Calculate the Euclidean distance of two center points
    dists = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=-1))

    # Calculate the distance between the groundtruth center point and the vertexz of the image
    thr_ul = np.sqrt(np.power(centers2[..., 0], 2)+np.power(centers2[..., 1], 2)) # Upper left
    thr_ur = np.sqrt(np.power((width-centers2[..., 0]), 2)+np.power(centers2[..., 1], 2)) # Upper right
    thr_ll = np.sqrt(np.power(centers2[..., 0], 2)+np.power((height-centers2[..., 1]), 2)) # Lower left
    thr_lr = np.sqrt(np.power((width-centers2[..., 0]), 2)+np.power((height-centers2[..., 1]), 2)) # Lower right

    def calculate_dist(point1, point2):
        return np.sqrt(np.power(point1[0]-point2[0],2)+np.power(point1[1]-point2[1],2))

    def calculate_detla(point, gt):
        # judge the prediction box center point with groundtruth

        # the center point of prediction box
        box_cx = point[0]
        box_cy = point[1]
        # the groundtruth four points information
        gt_xmin = gt[0]
        gt_ymin = gt[1]
        gt_xmax = gt[2]+gt[0]
        gt_ymax = gt[3]+gt[1]

        # flag calculates the points in area 5 (groundtruth box)
        flag = False

        # delta represents the shortest distence for center point of prediction box with the groundtruth boundary

        # judge in area 1 (upper left area)
        # for area 1, delta represents the distence for center point of prediction box with the upper left vertex of groundtruth box
        if box_cx <= gt_xmin and box_cy <= gt_ymin:
            delta = calculate_dist(point, (gt_xmin, gt_ymin))
        # judge in area 2 (upper area)
        # for area 2, delta represents the distence for center point of prediction box with the upper edge of groundtruth box
        if (gt_xmin < box_cx and box_cx <= gt_xmax) and box_cy <= gt_ymin:
            delta = gt_ymin - box_cy
        # judge in area 3 (upper right area)
        # for area 3, delta represents the distence for center point of prediction box with the upper right vertex of groundtruth box
        if gt_xmax < box_cx  and box_cy <= gt_ymin:
            delta = calculate_dist(point, (gt_xmax, gt_ymin))
        # judge in area 4 (left area)
        # for area 4, delta represents the distence for center point of prediction box with the left edge of groundtruth box
        if box_cx <= gt_xmin and (gt_ymin < box_cy and box_cy <= gt_ymax):
            delta = gt_xmin - box_cx
        # judge in area 5 (groundtruth box)
        # for area 5, delta is 0 since the center point of prediction box locates in the groundtruth box
        if (gt_xmin < box_cx and box_cx <= gt_xmax) and (gt_ymin < box_cy and box_cy <= gt_ymax):
            delta = 0
            flag = True
        # judge in area 6 (right area)
        # for area 6, delta represents the distence for center point of prediction box with the right edge of groundtruth box
        if gt_xmax < box_cx and (gt_ymin < box_cy and box_cy <= gt_ymax):
            delta = box_cy - gt_ymax
        # judge in area 7 (lower left area)
        # for area 7, delta represents the distence for center point of prediction box with the lower left vertex of groundtruth box
        if box_cx <= gt_xmin and gt_ymax < box_cy:
            delta = calculate_dist(point, (gt_xmin, gt_ymax))
        # judge in area 8 (lower area)
        # for area 8, delta represents the distence for center point of prediction box with the lower edge of groundtruth box
        if (gt_xmin < box_cx and box_cx <= gt_xmax) and gt_ymax < box_cy:
            delta = box_cy - gt_ymax
        # judge in area 9 (lower right area)
        # for area 9, delta represents the distence for center point of prediction box with the lower right vertex of groundtruth box
        if gt_xmax < box_cx and gt_ymax < box_cy:
            delta = calculate_dist(point, (gt_xmax, gt_ymax))

        return delta, flag
    
    def normalization(min, max, num):
        return (num-min)/(max-min)
            
    errors = np.zeros(len(rects1))
    flags = np.zeros(len(rects1))

    for i in range(len(rects1)):
        delta, flag = calculate_detla(centers1[i], rects2[i])

        # sum the points in groundtruth area
        if flag == False:
            flags[i] = 0
        else:
            flags[i] = 1
        
        # add the delta value as penalty factor
        error = dists[i] + delta
        
        # the max error is the distence for center point of groundtrut box with one of the four vertex in existing frame 
        thr_max = max((thr_ul[i]+calculate_detla((0, 0), rects2[i])[0]), (thr_ur[i]+calculate_detla((width, 0), rects2[i])[0]), (thr_ll[i]+calculate_detla((0, height), rects2[i])[0]), (thr_lr[i]+calculate_detla((width, height), rects2[i])[0]))

        # use the max value as threshold and normalize the error value
        error = normalization(0, thr_max, error)
        errors[i] = error
    return errors, flags







