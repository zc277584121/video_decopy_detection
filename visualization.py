import json
import os
from typing import List
import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def visual_one_np(np_path, save_dir='./visual_imgs', gt_box_list: List[List] = None, pred_box_list: List[List] = None,
                  fps=1):
    video_feature = np.load(np_path)
    np_file_name = np_path.split(os.path.sep)[-1][:-4]
    print('video_feature.shape = ', video_feature.shape)
    plt.figure()
    plt.imshow(video_feature)

    ax = plt.gca()
    if gt_box_list is not None:
        for gt_box in gt_box_list:
            gt_box = [t * fps for t in gt_box]
            y1 = gt_box[0]
            x1 = gt_box[1]
            y2 = gt_box[2]
            x2 = gt_box[3]
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            rect = patches.Rectangle((x1, y1),
                                     width=width,
                                     height=height,
                                     linewidth=1,
                                     edgecolor='cyan',
                                     fill=False)

            ax.add_patch(rect)
    if pred_box_list is not None:
        for pred_box in pred_box_list:
            pred_box = [t * fps for t in pred_box]
            y1 = pred_box[0]
            x1 = pred_box[1]
            y2 = pred_box[2]
            x2 = pred_box[3]
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            rect = patches.Rectangle((x1, y1),
                                     width=width,
                                     height=height,
                                     linewidth=1,
                                     edgecolor='red',
                                     fill=False)

            ax.add_patch(rect)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, np_file_name + '.png')
    plt.savefig(save_path)


def visual_np_files(np_folder,
                    save_dir='./visual_imgs',
                    gt_file=None,
                    pred_file=None,
                    fps=1,
                    sample_num=None):
    random.seed(42)
    gt_dict = json.load(open(gt_file)) if gt_file is not None else None
    pred_dict = json.load(open(pred_file)) if pred_file is not None else None
    np_path_list = [os.path.join(np_folder, np_file) for np_file in os.listdir(np_folder)]
    if sample_num is not None:
        np_path_list = random.sample(np_path_list, sample_num)

    for np_path in np_path_list:
        np_file_name = np_path.split(os.path.sep)[-1][:-4]
        gt_box_list = gt_dict[np_file_name] if gt_file is not None else None
        pred_box_list = pred_dict[np_file_name] if pred_file is not None else None
        visual_one_np(np_path, save_dir=save_dir, gt_box_list=gt_box_list, pred_box_list=pred_box_list, fps=fps)


if __name__ == '__main__':
    # visual_one_np(
    #     '/home/zhangchen/zhangchen_workspace/video_decopy_detection/sim_matrix_npy/vcsl-dns_backbone-val_pairs-dns_sim/ffc548adbfe24a79b6ab7f444e2a8b98-fcb6fc1ea0984b43a96fb4296e0ae03f.npy',
    #     save_dir='/home/zhangchen/zhangchen_workspace/video_decopy_detection/visual_imgs',
    #     gt_box_list=[[18, 6, 57, 45]])

    # visual_np_files('/home/zhangchen/zhangchen_workspace/video_decopy_detection/sim_matrix_npy/vcsl-dns_backbone-val_pairs-dns_sim/',
    #                 gt_file='./vcsl_data/label_file_uuid_total.json',
    #                 sample_num=10)

    visual_np_files('/home/zhangchen/zhangchen_workspace/video_decopy_detection/sim_matrix_npy/muscle-dns_backbone-qd_pair-dns_sim',
                    pred_file='./result/best_pred/muscle-dns_backbone-st2_pairs-dns_sim-DTW-pred.json',
                    save_dir='/home/zhangchen/zhangchen_workspace/video_decopy_detection/visual_imgs/muscle-dns_backbone-st2_pairs-dns_sim-DTW')