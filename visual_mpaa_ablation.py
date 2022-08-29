import argparse
import os
from collections import defaultdict
from pprint import pprint

from matplotlib import pyplot as plt

from datasets import MPAA


def visual_filter_thresh_ablation(dataset, pred_file, output_path, start=1, end=180, interval=10):
    trick_name_list = list(dataset.trick_format_dict.keys()) + list(dataset.special_trick_dict.keys()) + ['all']
    name_2_ys = defaultdict(list)
    for filter_thresh in range(start, end, interval):
        case_res_dict = dataset.evaluate(pred_file, 'f1', filter_thresh=filter_thresh)
        for trick_name, trick_res in case_res_dict.items():
            name_2_ys[trick_name].append(trick_res['f1'])
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for trick_name in trick_name_list:
        plt.figure()
        plt.plot(list(range(start, end, interval)), name_2_ys[trick_name], color='r',marker='o',linestyle='-.',alpha=0.5)
        plt.savefig(os.path.join(output_path, trick_name + '.png'))
    max_f1_dict = dict()
    for trick_name, ys in name_2_ys.items():
        max_f1_dict[trick_name] = max(ys)
    pprint(max_f1_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="result dir of video segment prediction")
    parser.add_argument("--video_root", type=str, default=None, help="Useful only when MPAA")
    parser.add_argument("--output_path", type=str, default=None, help="output path.")

    args = parser.parse_args()

    dataset = MPAA(video_root=args.video_root)
    visual_filter_thresh_ablation(dataset, args.pred_file, args.output_path)

### python visual_mpaa_ablation.py --pred_file ./result/default_pred/mpaa-dns_backbone-qd_pair-dns_sim-TN-pred.json --video_root /home/zhangchen/zhangchen_workspace/dataset/frank/MPAA --output_path ./visual_mpaa_ablation

