#!/usr/bin/env python3
# Copyright (c) Ant Group, Inc.
"""
Codes for [CVPR2022] VCSL paper [https://github.com/alipay/VCSL].
This is the script for evaluating the performance of video copy detection algorithm result.
Before running this script, you need to go to run_video_sim.py and run_video_vta.py files
to get similarity map and obtain copied segments (i.e., temporal alignment results).
This script will give the following three evaluation metrics:
- Overall segment-level precision/recall performance
- Video-level FRR/FAR performance
- Segment-level precision/recall performance on positive samples over query set
We recommend using the first metric to reflect segment-level alignment accuracy, while it
is also influenced by video-level results.

Please cite the following publications if you plan to use our codes or the results for your research:
{
    1. He S, Yang X, Jiang C, et al. A Large-scale Comprehensive Dataset and Copy-overlap Aware Evaluation
    Protocol for Segment-level Video Copy Detection[C]//Proceedings of the IEEE/CVF Conference on Computer
    Vision and Pattern Recognition. 2022: 21086-21095.
    2. Jiang C, Huang K, He S, et al. Learning segment similarity and alignment in large-scale content based
    video retrieval[C]//Proceedings of the 29th ACM International Conference on Multimedia. 2021: 1618-1626.
}
@author: Sifeng He and Xudong Yang
@email [sifeng.hsf@antgroup.com, jiegang.yxd@antgroup.com]
"""


import argparse
import json
import os

import numpy as np
import pandas as pd
from multiprocessing import Pool

from utils import build_reader
from vcsl.metric import evaluate_micro, evaluate_macro, precision_recall


def run_eval(input_dict):
    gt_box = np.array(input_dict["gt"])
    pred_box = np.array(input_dict["pred"])
    result_dict = precision_recall(pred_box, gt_box)
    result_dict["name"] = input_dict["name"]
    return result_dict


if __name__ == '__main__':
    #python evaluate.py --pred_file ./result/output/dino-DTW-pred.json --split test
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_file", default="vcsl_data/label_file_uuid_total.json", type=str, help="gt label file")
    parser.add_argument("--pair_group_file", default="vcsl_data/split_meta_pairs.json",
                        type=str, help="meta pair corresponding relation")
    parser.add_argument("--meta_info", default="vcsl_data/meta_info.json",
                        type=str, help="meta info with name and categories")
    parser.add_argument("--pred_file", type=str, help="result dir of video segment prediction")
    parser.add_argument("--split", type=str, default='test', help="store of pred data: train|val|test|all")
    parser.add_argument("--local_set_csv", type=str, help="save result csv file with query set, endswith '.csv' ")
    parser.add_argument("--pool_size", type=int, default=16, help="multiprocess pool size of evaluation")
    parser.add_argument("--ratio_pos_neg", type=float, default=1, help="ratio between positive and negative samples")

    args = parser.parse_args()

    split = args.split if args.split is not None else "all"
    if args.split not in ['all', 'train', 'val', 'test']:
        raise ValueError(f"Unknown dataset split {args.split}, must be one of train|val|test")

    config = dict()
    reader = build_reader('local', "json", **config)

    print(f"start loading...")

    gt = json.load(open(args.anno_file))
    key_list = [key for key in gt]

    meta_pairs = json.load(open(args.pair_group_file))

    root_dir = os.path.dirname(args.anno_file)
    split_file = os.path.join(root_dir, f"pair_file_{args.split}.csv")
    df = pd.read_csv(split_file)
    split_pairs = set([f"{q}-{r}" for q, r in zip(df.query_id.values, df.reference_id.values)])
    print("{} contains pairs {}", args.split, len(split_pairs))

    key_list = [key for key in key_list if key in split_pairs]
    print("Copied video data (positive) to evaluate: {}", len(key_list))

    pred_dict = reader.read(args.pred_file)
    eval_list = []
    for key in split_pairs:
        if key in gt:
            eval_list += [{"name": key, "gt": gt[key], "pred": pred_dict[key]}]
        else:
            eval_list += [{"name": key, "gt": [], "pred": pred_dict[key]}]

    print(f"finish loading files, start evaluation...")

    process_pool = Pool(args.pool_size)
    result_list = process_pool.map(run_eval, eval_list)
    print('result_list[0] =', result_list[0])
    result_dict = {i['name']: i for i in result_list}

    if args.split != 'all':
        meta_pairs = meta_pairs[args.split]
    else:
        meta_pairs = {**meta_pairs['train'], **meta_pairs['val'], **meta_pairs['test']}

    try:
        feat, vta = args.pred_file.split('-')[:2]
    except:
        feat, vta = 'My-FEAT', 'My-VTA'

    # Evaluated result on all video pairs including positive and negative copied pairs.
    # The segment-level precision/recall can also indicate video-level performance since
    # missing or false alarm lead to decrease on segment recall or precision.
    r, p, frr, far = evaluate_micro(result_dict, args.ratio_pos_neg)
    print(f"Feature {feat} & VTA {vta}: ")
    print(f"Overall segment-level performance, "
                f"Recall: {r:.2%}, "
                f"Precision: {p:.2%}, "
                f"F1: {2 * r * p / (r + p):.2%}, "
                )
    print(f"video-level performance, "
                f"FRR: {frr:.2%}, "
                f"FAR: {far:.2%}, "
                )

    # Evaluated macro result over each query set.
    # print('list(meta_pairs.keys())[0] =', list(meta_pairs.keys())[0])
    # print('list(meta_pairs.values())[0] =', list(meta_pairs.values())[0])
    r, p, cnt = evaluate_macro(result_dict, meta_pairs)

    print(f"query set cnt {cnt}, "
                f"query macro-Recall: {r:.2%}, "
                f"query macro-Precision: {p:.2%}, "
                f"F1: {2 * r * p / (r + p):.2%}, "
                )

