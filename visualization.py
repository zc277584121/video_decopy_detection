import os

import numpy as np
from matplotlib import pyplot as plt


def visual_one_np(np_path, save_dir='./visual_imgs'):
    video_feature = np.load(np_path)
    np_file_name = np_path.split(os.path.sep)[-1].split('.')[0]
    print('video_feature.shape = ', video_feature.shape)
    plt.figure()
    plt.imshow(video_feature)
    save_path = os.path.join(save_dir, np_file_name + '.png')
    plt.savefig(save_path)

if __name__ == '__main__':
    visual_one_np('/home/zhangchen/zhangchen_workspace/video_decopy_detection/sim_matrix_npy-without_pairs-dns_sim/281bd7501b1b4ce2ab424c724f0931ff-b7dc5f07783c4784858582ceb78a402d.npy',
                  save_dir='/home/zhangchen/zhangchen_workspace/video_decopy_detection/visual_imgs')
