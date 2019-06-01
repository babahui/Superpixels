import os
from os.path import basename, join, isfile
from imageio import imread, imwrite
from scipy.io import loadmat
from skimage.segmentation import find_boundaries
import numpy as np

# read
def evaluate(img_dir, gt_dir, soft_thres=1):

    img_list = [join(img_dir, f) for f in os.listdir(img_dir) if isfile(join(img_dir, f))]
    precision_array, recall_array, F_array = [], [], []

    for img_path in img_list:
        if img_path[-3:] == 'jpg':
#             print(img_path)
            segment_img = imread(img_path)
#             print(segment_img.dtype)
            
            # get .mat name and convert to numpy type
            gt_name = img_path.split('/')[-1].split('.')[0] + '.mat'
            gt_path = join(gt_dir, gt_name)

            if isfile(gt_path):
                gt_dict = loadmat(gt_path)
            else:
                print('no .mat files')

            gt_mat = gt_dict['groundTruth']
            
            
            # evaluate p, r, F, return rank-1
            best_precision, best_recall, best_F = 0, 0, 0
            y_pred = find_boundaries(label_img=segment_img, connectivity=1, mode='thick').astype(np.uint8)
            for i in range(gt_mat.shape[1]):     # in 6 groudtruth situation
                ture_bound = gt_mat[0, i][0, 0][0]
                y_ture = find_boundaries(label_img=ture_bound, connectivity=1, mode='thick').astype(np.uint8)
                precision, recall, F = measurement(y_ture, y_pred, soft_thres=soft_thres)
           
                if best_precision < precision:
                    best_precision = precision
                if best_recall < recall:
                    best_recall = recall
                if best_precision + best_recall > 0:
                    best_F = (2 * best_precision * best_recall) / (best_precision + best_recall)    
                else:
                    best_F = 0
                    
            precision_array.append(best_precision)
            recall_array.append(best_recall)
            F_array.append(best_F)       
#             print('first time p, r, F: ',precision_array, recall_array, F_array)

    return np.asarray(precision_array), np.asarray(recall_array), np.asarray(F_array)
            
    
def measurement(y_ture, y_pred, soft_thres):
    # make sure soft threshold not too enough to destory our method
    if soft_thres > 100:
        print('maybe too large soft_thres param setting')
    import numpy as np
    y_ture = np.asarray(y_ture)
    y_pred = np.asarray(y_pred)
    
    fill_len = 100
    fill_y_ture = np.zeros((2 * fill_len + y_ture.shape[0], 2 * fill_len + y_ture.shape[1]))
    fill_y_pred = np.zeros((2 * fill_len + y_pred.shape[0], 2 * fill_len + y_pred.shape[1]))

    fill_y_ture[fill_len:(fill_len + y_ture.shape[0]), fill_len:(fill_len + y_ture.shape[1])] = y_ture
    fill_y_pred[fill_len:(fill_len + y_pred.shape[0]), fill_len:(fill_len + y_pred.shape[1])] = y_pred
    
    tp, fp, fn, tn= 0, 0, 0, 0
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            if y_pred[i, j] == 1:
                # if found element 1 in y_ture region
                if 1 in fill_y_ture[i+fill_len-soft_thres:i+fill_len+soft_thres+1, j+fill_len-soft_thres:j+fill_len+soft_thres+1]:
                    tp += 1
                else:
                    tn += 1
                    
            elif y_pred[i, j] == 0 and y_ture[i, j] == 1:
                if 1 not in fill_y_pred[i+fill_len-soft_thres:i+fill_len+soft_thres+1, j+fill_len-soft_thres:j+fill_len+soft_thres+1]:
                    fp += 1
            elif y_pred[i, j] == 0 and y_ture[i, j] == 0:
                fn += 1
                
#     print('tp, tn, fp, fn', tp, tn, fp, fn)
    
    recall = tp / (tp + fp)
    precision = tp / (tp + tn)
    if precision == 0 and recall == 0:
        F = 0
    else:
        F = 2 * (precision * recall) / (precision + recall)
    return precision, recall, F