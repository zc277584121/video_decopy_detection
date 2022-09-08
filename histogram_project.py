import math
import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import skimage.measure
import time
import random

random.seed(42)

def rotate_bound(image, angle) -> np.ndarray:
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return cv2.warpAffine(image, M, (nW, nH))

def histogram_project(sim_matrix, angle, peak_rate_thresh=0.85, peak_width_thresh=6, accumulate_rate_thresh=0.4, matrix_h_thresh=100, reduce_size=4):
    src_h, src_w = sim_matrix.shape[0], sim_matrix.shape[1]
    if reduce_size > 1:
        peak_width_thresh = peak_width_thresh // reduce_size
        sim_matrix = skimage.measure.block_reduce(sim_matrix, (reduce_size, reduce_size), np.max)
        # sim_matrix = skimage.measure.block_reduce(sim_matrix, (reduce_size, reduce_size), np.mean)
    # sim_matrix = np.load(sim_npy_file)
    # print('sim_matrix max = ', sim_matrix.max())
    # print('sim_matrix min = ', sim_matrix.min())
    sim_matrix = (sim_matrix - sim_matrix.min())/(sim_matrix.max() - sim_matrix.min())



    # print('sim_matrix max = ', sim_matrix.max())
    # print('sim_matrix min = ', sim_matrix.min())

    rotated_matrix = rotate_bound(sim_matrix, angle)


    project_sum = rotated_matrix.sum(axis=0)
    # print('rotated_matrix.shape = ', rotated_matrix.shape)
    # print('project_sum.shape = ', project_sum.shape)

    height, width = rotated_matrix.shape[0], rotated_matrix.shape[1]
    # blankImage = np.zeros_like(rotated_matrix)
    # Make the vertical projection histogram

    max_x, max_y = project_sum.argmax(axis=0), project_sum.max(axis=0)
    peak_height_thresh = max_y * peak_rate_thresh
    # ascend_x_list = []
    # descend_x_list = []
    # for x in range(1, width - 1):
    #     if project_sum[x - 1] < peak_height_thresh <= project_sum[x]:
    #         ascend_x_list.append(x)
    #     elif project_sum[x - 1] >= peak_height_thresh > project_sum[x]:
    #         descend_x_list.append(x)
    # find_x_list = []
    # for ascend_x in ascend_x_list:
    #     tmp_descend_x_list = sorted(descend_x_list + [ascend_x])
    #     next_descend_x = tmp_descend_x_list.index(ascend_x) + 1
    #     if next_descend_x >= len(tmp_descend_x_list):
    #         continue
    #     if next_descend_x - ascend_x < peak_width_thresh:
    #         find_x_list.append((next_descend_x + ascend_x) / 2)
    find = []
    peak_left_x = int(max(0, max_x - peak_width_thresh))
    peak_right_x = int(min(max_x + peak_width_thresh, width - 1))
    # print('peak_left_x =', peak_left_x)
    # print('peak_right_x =', peak_right_x)
    accumulate_max_h = src_h / math.sin(math.radians(angle)) / reduce_size
    accumulate_rate = max_y / accumulate_max_h

    # print('project_sum[peak_left_x] = ', project_sum[peak_left_x])
    # print('project_sum[peak_right_x] = ', project_sum[peak_right_x])
    # print('peak_height_thresh = ', peak_height_thresh)
    # print('accumulate_rate = ', accumulate_rate)
    # print('accumulate_rate_thresh = ', accumulate_rate_thresh)
    # print('matrix_h_thresh = ', matrix_h_thresh)
    # print('src_h = ', src_h)
    if project_sum[peak_left_x] < peak_height_thresh and project_sum[peak_right_x] < peak_height_thresh \
            and accumulate_rate > accumulate_rate_thresh \
            and src_h > matrix_h_thresh:
        find.append([0, 0, src_h, src_w])

    # save fig
    # for idx, value in enumerate(project_sum):
    #     cv2.line(rotated_matrix, (idx, height), (idx, height - int(value)), (1, 1, 1), 1)
    # # cv2.line(rotated_matrix, (max_x, 0), (max_x, height), (0.5, 0.5, 0.5), 1) # peak_line
    # cv2.line(rotated_matrix, (peak_left_x, 0), (peak_left_x, height), (0.5, 0.5, 0.5), 1) # peak_line
    # cv2.line(rotated_matrix, (peak_right_x, 0), (peak_right_x, height), (0.5, 0.5, 0.5), 1) # peak_line
    # cv2.line(rotated_matrix, (0, height - int(peak_height_thresh)), (width, height - int(peak_height_thresh)), (0.1, 0.1, 0.1), 1) # thresh_line
    # plt.figure(figsize=(65, 65))
    # plt.imshow(rotated_matrix)
    # plt.savefig('./tmp_img.png')
    ## plt.show()
    return find


def detect_by_histogram_project(sim_matrix, angle_min=40, angle_max=50, interval=1, peak_rate_thresh=0.85, peak_width_thresh=6, accumulate_rate_thresh=0.4, matrix_h_thresh=100, reduce_size=4):
    # max_find = 0
    # max_find_angle = -1
    find = []
    # tmp_rate = -1
    for angle in range(angle_min, angle_max, interval):
        find = histogram_project(sim_matrix, angle, peak_rate_thresh, peak_width_thresh, accumulate_rate_thresh, matrix_h_thresh, reduce_size)
        if find:
            print(f'angle {angle} find!')
            break
    return find

