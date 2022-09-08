import argparse
import json
import os
import random
from functools import partial
from multiprocessing import Pool
from typing import Dict

# import h5py
import numpy as np
import pandas as pd

from datasets import filter_short_pred, MPAA
from utils import Reader, AsyncWriter, DataType, build_writer, build_reader
from torch.utils.data import DataLoader
from loguru import logger
from itertools import islice, product

from vcsl.datasets import ItemDataset, filter_pair_list_by_np_dir
from vcsl.metric import evaluate_macro, precision_recall
from vcsl.vta import build_vta_model


def parse_range(arg_str):
    tokens = arg_str.strip().split(':')
    if len(tokens) == 2:
        start, end = map(int, tokens)
        return list(range(start, end))
    elif len(tokens) == 3:
        start, end, step = map(float, tokens)
        return np.arange(start, end, step).tolist()
    elif len(tokens) == 1:
        return [float(tokens[0])]


def gen_input(key: str, reader: Reader, root: str, gt: Dict):
    return {"name": key,
            "gt": gt[key],
            "pred": json.loads(reader.read(os.path.join(root, key + ".json")))}


def run_eval(input_dict):
    gt_box = np.array(input_dict["gt"])
    pred_box = np.array(input_dict["pred"])
    result_dict = precision_recall(pred_box, gt_box)
    result_dict["name"] = input_dict["name"]
    return result_dict


def filter_partial_data(data_list, baseline_predict_file, gt_json_file):
    pred_dict = json.load(open(baseline_predict_file))
    gt_dict = json.load(open(gt_json_file))
    FN_list = []
    TN_list = []
    TP_list = []
    FP_list = []
    for pred_pair_str, pred_value in pred_dict.items():
        if pred_value is None or len(pred_value) == 0 \
                or filter_short_pred(pred_value, filter_thresh=args.filter_thresh):
            if len(gt_dict[pred_pair_str]) > 0:
                FN_list.append(pred_pair_str)
            else:
                TN_list.append(pred_pair_str)
        else:
            if len(gt_dict[pred_pair_str]) > 0:
                TP_list.append(pred_pair_str)
            else:
                FP_list.append(pred_pair_str)
    TN_sample_num = min(max(len(FN_list), len(TP_list), len(FP_list)) * 10, len(TN_list))
    TN_sample_list = random.sample(TN_list, k=TN_sample_num)
    filtered_all_pairs = FN_list + TN_sample_list + TP_list + FP_list
    filtered_data_list = [data for data in data_list if data[0] in filtered_all_pairs]

    print(f'before filter num ={len(data_list)},'
          f'after filter num ={len(filtered_data_list)}')
    return filtered_data_list


def hyper_params_search(args):
    pairs, files_dict, query, reference = None, None, None, None
    if args.pair_file:
        df = pd.read_csv(args.pair_file)
        pairs = df[['query_id', 'reference_id']].values.tolist()

        pairs = filter_pair_list_by_np_dir(pairs, args.input_root)

        data_list = [(f"{p[0]}-{p[1]}", f"{p[0]}-{p[1]}") for p in pairs]
    elif args.query_file and args.reference_file:
        query = pd.read_csv(args.query_file)
        query = query[['uuid']].values.tolist()

        reference = pd.read_csv(args.reference_file)
        reference = reference[['uuid']].values.tolist()

        pairs = product(query, reference)
        data_list = [(f"{p[0]}-{p[1]}", f"{p[0]}-{p[1]}") for p in pairs]
    else:
        data_list = []
        sim_npy_files = os.listdir(args.input_root)
        for sim_npy_file in sim_npy_files:
            file_name = sim_npy_file[:-4]
            if args.dataset == 'MPAA':
                file_ext_set = {'wmv', 'flv', 'out', 'MPG', 'vog', 'mpg', 'mov', 'VOB', 'avi', 'sh', 'txt', 'vob',
                                'rmvb', 'mpeg', 'mp4'}
                split_plan_num = 0
                for file_ext in file_ext_set:
                    if len(file_name.split(f'.{file_ext}-')) == 2:
                        query_id = file_name.split(f'.{file_ext}-')[0] + f'.{file_ext}'
                        ref_id = file_name.split(f'.{file_ext}-')[1]
                        split_plan_num += 1
                if split_plan_num != 1:
                    raise RuntimeError(f'{sim_npy_file} can not split well.')
            else:
                query_id = file_name.split('-')[0]
                ref_id = file_name.split('-')[1]
            data_list.append((f"{query_id}-{ref_id}", f"{query_id}-{ref_id}"))

    if args.use_partial_data:
        data_list = filter_partial_data(data_list, args.baseline_predict_file, args.gt_json_file)

    dataset = ItemDataset(data_list,
                          store_type='local',
                          data_type=DataType.NUMPY.type_name,
                          root=args.input_root,
                          trans_key_func=lambda x: x + '.npy',
                          use_cache=False,
                          )

    logger.info(f"Data to run {len(dataset)}")

    loader = DataLoader(dataset, collate_fn=lambda x: x,
                        batch_size=args.batch_size,
                        num_workers=args.data_workers)

    tn_max_step = map(int, parse_range(args.tn_max_step))
    tn_top_k = map(int, parse_range(args.tn_top_K))
    min_sim = parse_range(args.min_sim)
    max_path = map(int, parse_range(args.max_path))
    min_length = map(int, parse_range(args.min_length))
    max_iou = parse_range(args.max_iou)


    angle_min = map(int, parse_range(args.angle_min))
    angle_max = map(int, parse_range(args.angle_max))
    interval = map(int, parse_range(args.interval))
    peak_rate_thresh = parse_range(args.peak_rate_thresh)
    peak_width_thresh = parse_range(args.peak_width_thresh)
    accumulate_rate_thresh = parse_range(args.accumulate_rate_thresh)
    matrix_h_thresh = parse_range(args.matrix_h_thresh)
    reduce_size = map(int, parse_range(args.reduce_size))



    discontinue = map(int, parse_range(args.discontinue))
    sum_sim = parse_range(args.sum_sim)
    ave_sim = parse_range(args.ave_sim)
    diagonal_thres = map(int, parse_range(args.diagonal_thres))

    if args.alignment_method.startswith('DTW'):
        hyper_params = [dict(discontinue=values[0],
                             min_sim=values[1],
                             min_length=values[2],
                             max_iou=values[3])
                        for values in product(discontinue, min_sim, min_length, max_iou)
                        ]

    elif args.alignment_method == 'TN':
        hyper_params = [dict(tn_max_step=values[0],
                             tn_top_k=values[1],
                             max_path=values[2],
                             min_sim=values[3],
                             min_length=values[4],
                             max_iou=values[5])
                        for values in product(tn_max_step, tn_top_k, max_path, min_sim, min_length, max_iou)]

    elif args.alignment_method.startswith('DP'):
        hyper_params = [dict(discontinue=values[0],
                             min_sim=values[1],
                             ave_sim=values[2],
                             min_length=values[3],
                             diagonal_thres=values[4])
                        for values in product(discontinue, min_sim, ave_sim, min_length, diagonal_thres)
                        ]

    elif args.alignment_method.startswith('HV'):
        hyper_params = [dict(min_sim=values[0], iou_thresh=values[1])
                        for values in product(min_sim, max_iou)
                        ]

    elif args.alignment_method == 'PM':
        hyper_params = [dict(angle_min=values[0],
                             angle_max=values[1],
                             interval=values[2],
                             peak_rate_thresh=values[3],
                             peak_width_thresh=values[4],
                             accumulate_rate_thresh=values[5],
                             matrix_h_thresh=values[6],
                             reduce_size=values[7])
                        for values in product(angle_min,
                                              angle_max,
                                              interval,
                                              peak_rate_thresh,
                                              peak_width_thresh,
                                              accumulate_rate_thresh,
                                              matrix_h_thresh,
                                              reduce_size)]

    elif args.alignment_method == 'TN+PM':
        hyper_params = [dict(tn_max_step=values[0],
                             tn_top_k=values[1],
                             max_path=values[2],
                             min_sim=values[3],
                             min_length=values[4],
                             max_iou=values[5],
                             angle_min=values[6],
                             angle_max=values[7],
                             interval=values[8],
                             peak_rate_thresh=values[9],
                             peak_width_thresh=values[10],
                             accumulate_rate_thresh=values[11],
                             matrix_h_thresh=values[12],
                             reduce_size=values[13])
                        for values in product(tn_max_step, tn_top_k, max_path, min_sim, min_length, max_iou,
                                              angle_min,
                                              angle_max,
                                              interval,
                                              peak_rate_thresh,
                                              peak_width_thresh,
                                              accumulate_rate_thresh,
                                              matrix_h_thresh,
                                              reduce_size)]

    else:
        raise ValueError(f"Unknown VTA method: {args.alignment_method}")

    output_store = 'local'
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root, exist_ok=True)

    writer_pool = AsyncWriter(pool_size=args.output_workers,
                              store_type=output_store,
                              data_type=DataType.JSON.type_name,
                              )

    writer_pool.consume((os.path.join(args.output_root, f"hyper_params.json"), hyper_params))

    for i, model_config in enumerate(hyper_params):
        logger.info("hyper params {}: {}", i, str(model_config))

        model = build_vta_model(method=args.alignment_method, concurrency=args.request_workers, **model_config)

        sub_dir = f"{i}"
        # if output_store == 'local':
        os.makedirs(os.path.join(args.output_root, sub_dir), exist_ok=True)

        total_result = dict()
        ii = 0
        for batch_data in islice(loader, 0, None):
            logger.info(f"{ii} / {len(loader)}, {i} / {len(hyper_params)}")
            batch_result = model.forward_sim(batch_data)
            # logger.info("result cnt: {}", len(batch_result))
            ii += 1
            for pair_id, result in batch_result:
                total_result[pair_id] = result

        output_sub = os.path.join(args.output_root, sub_dir)
        if not os.path.exists(output_sub):
            os.makedirs(output_sub, exist_ok=True)
        writer = build_writer(output_store, DataType.JSON.type_name)
        writer.write(os.path.join(output_sub, 'predict_json.json'), total_result)

    writer_pool.stop()

#
# def eval_all(args):
#     gt = json.load(open(args.anno_file))
#     key_list = [key for key in gt]
#     root_dir = os.path.dirname(args.anno_file)
#     split_file = os.path.join(root_dir, f"pair_file_val.csv")
#     df = pd.read_csv(split_file)
#     # split_pairs = set([f"{q}-{r}" for q, r in zip(df.query_id.values, df.reference_id.values)])
#
#     split_pairs = df[['query_id', 'reference_id']].values.tolist()
#     split_pairs = filter_pair_list_by_np_dir(split_pairs, args.input_root)
#     split_pairs = set([f"{q_r[0]}-{q_r[1]}" for q_r in split_pairs])
#
#     logger.info("Val set contains pairs {}", len(split_pairs))
#
#     key_list = [key for key in key_list if key in split_pairs]
#     print('len(gt intersection split file) =', len(key_list))
#
#     pair_group_dict = json.load(open(args.pair_group_file))['val']
#
#     print('len(query file in pair_group_file)=', len(pair_group_dict))
#
#     output_store = 'local'
#     reader = build_reader(output_store, "bytes")
#
#     process_pool = Pool(args.data_workers)
#
#     final_results = {'results': [], 'best': None}
#     best, best_idx = dict(result=dict(f1=-1)), -1
#
#     hyper_params = json.loads(reader.read(os.path.join(args.output_root, 'hyper_params.json')))
#     for idx, param in enumerate(islice(hyper_params, 0, None)):
#         data_root = os.path.join(args.output_root, str(idx))
#
#         read_func = partial(gen_input, reader=reader, root=data_root, gt=gt)
#         eval_list = process_pool.map(read_func, key_list)
#
#         logger.info(f"finish loading files, start evaluation...")
#
#         result_list = process_pool.map(run_eval, eval_list)
#         result_dict = {i['name']: i for i in result_list}
#         r, p, cnt = evaluate_macro(result_dict, pair_group_dict)
#         f1 = 2 * r * p / (r + p)
#         logger.info(f"Hyper params {idx}: {param}")
#         logger.info(
#             f"query set cnt {cnt}, "
#             f"query macro-Recall: {r:.2%}, "
#             f"query macro-Precision: {p:.2%}, "
#             f"F1: {f1:.2%}")
#
#         result = dict(param=param, result=dict(r=r, p=p, f1=f1))
#         best, best_idx = (result, idx) \
#             if f1 > best['result']['f1'] else (best, best_idx)
#         final_results['results'].append(result)
#         logger.info('')
#
#     final_results['best'] = best
#     final_results['best_idx'] = best_idx
#     logger.info(f'best params {best_idx}: {best["param"]}')
#     logger.info(f'best result: {best["result"]}')
#
#     writer = build_writer(output_store, DataType.BYTES.type_name)
#     d = json.dumps(final_results, indent=2, ensure_ascii=False)
#     writer.write(os.path.join(args.output_root, 'result.json'), d)

def eval_all(args):
    hyper_param_list = json.load(open(os.path.join(args.output_root, 'hyper_params.json')))

    dataset = None
    if 'CC_WEB' in args.dataset:
        from datasets import CC_WEB_VIDEO

        dataset = CC_WEB_VIDEO()
    elif 'FIVR' in args.dataset:
        from datasets import FIVR

        dataset = FIVR(version=args.dataset.split('-')[1].lower())
    elif 'EVVE' in args.dataset:
        from datasets import EVVE

        dataset = EVVE()
    elif 'SVD' in args.dataset:
        from datasets import SVD

        dataset = SVD()
    elif 'VCSL' in args.dataset:
        from datasets import VCSL

        dataset = VCSL(datafolder='./vcsl_data', split=args.split)

    elif 'MPAA' in args.dataset:
        from datasets import MPAA
        dataset = MPAA(video_root=args.video_root)

    elif 'MUSCLE_VCD' in args.dataset:
        from datasets import MUSCLE_VCD
        dataset = MUSCLE_VCD(video_root='./muscle_vcd')

    best_param_ind = 0
    best_f1 = 0
    monitor_value = 'all' if args.monitor_value is None else args.monitor_value
    for i in range(len(hyper_param_list)):
        predict_json_file = os.path.join(args.output_root, str(i), 'predict_json.json')
        if args.filter_thresh is not None:
            res_dict = dataset.evaluate(predict_json_file, 'f1', filter_thresh=args.filter_thresh)
        else:
            res_dict = dataset.evaluate(predict_json_file, 'f1')
        if res_dict[monitor_value]['f1'] > best_f1:
            best_f1 = res_dict[monitor_value]['f1']
            best_param_ind = i
    print('*' * 80)
    print('best_param_ind =', best_param_ind)
    print('best_f1 =', best_f1)
    print('*' * 80)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--query_file", "-Q", type=str, help="data file")
    parser.add_argument("--reference_file", "-G", type=str, help="data file")
    parser.add_argument("--pair_file", type=str, help="data file")

    parser.add_argument("--dataset", type=str, help="dataset name, when using MPAA, please specify it, else not need.")
    parser.add_argument("--video_root", type=str, help="using MPAA, please specify it")

    # parser.add_argument("--input_store", type=str, help="store of input data: oss|local", default="oss")
    parser.add_argument("--input_root", type=str, help="root path of input data", default="")

    parser.add_argument("--use_partial_data", type=bool,
                        help="Due to the large amount of data, which takes a lot of time,"
                             "`use_partial_data` means use only a part of TN,"
                             "and it need specify params `baseline_predict_file` to get TN",
                        default=False)
    parser.add_argument("--baseline_predict_file", type=str,
                        help='baseline predict json file, only useful for `use_partial_data`')
    parser.add_argument("--gt_json_file", type=str,
                        help='ground truth json file, only useful for `use_partial_data`')

    parser.add_argument("--filter_thresh", type=float, default=0,
                        help="ratio between positive and negative samples, only useful for `use_partial_data`")

    parser.add_argument("--monitor_value", type=str, default=None)

    parser.add_argument("--batch_size", "-b", type=int, default=32, help="batch size")
    parser.add_argument("--data_workers", type=int, default=32, help="data workers")
    parser.add_argument("--request_workers", type=int, default=4, help="data workers")
    parser.add_argument("--output_workers", type=int, default=8, help="oss upload workers")
    parser.add_argument("--output_root", type=str, help="output root")
    # parser.add_argument("--output_store", type=str, help="store of input data: oss|local")

    # offline algorithm hyper parameters
    parser.add_argument("--alignment_method", type=str, default="DTW", help="DTW, DP, TN alignment method")
    parser.add_argument("--min_length", type=str, default="5", help="minimum length of one segment")
    parser.add_argument("--sum_sim", type=str, default="10.", help="minimum accumulated sim of one segment")
    parser.add_argument("--ave_sim", type=str, default="0.3", help="minimum accumulated sim of one segment")
    parser.add_argument("--min_sim", type=str, default="0.2", help="minimum average sim of one segment")
    parser.add_argument("--diagonal_thres", type=str, default="10", help="minimum average sim of one segment")
    parser.add_argument("--max_path", type=str, default="10", help="maximum path trials")
    parser.add_argument("--discontinue", type=str, default="3", help="max discontinue point in path")
    parser.add_argument("--tn_top_K", type=str, default="5", help="top k nearest in tn methods")
    parser.add_argument("--tn_max_step", type=str, default="10", help="max step in tn methods")
    parser.add_argument("--max_iou", type=str, default="0.3", help="max iou to filter bboxes")

    parser.add_argument("--angle_min", type=str, default="40", help="start angle for PM")
    parser.add_argument("--angle_max", type=str, default="50", help="end angle for PM")
    parser.add_argument("--interval", type=str, default="1", help="angle interval for PM")
    parser.add_argument("--peak_rate_thresh", type=str, default="0.85", help="peak rate thresh for PM")
    parser.add_argument("--peak_width_thresh", type=str, default="8", help="peak width thresh for PM")
    parser.add_argument("--accumulate_rate_thresh", type=str, default="0.4", help="accumulate rate thresh for PM")
    parser.add_argument("--matrix_h_thresh", type=str, default="100", help="matrix height thresh for PM")
    parser.add_argument("--reduce_size", type=str, default="4", help="reduce size for PM")

    parser.add_argument("--result_prefix", type=str, help="result path")

    parser.add_argument("--anno_file", default="./vcsl_data/label_file_uuid_total.json", type=str, help="gt label file")
    parser.add_argument("--pair_group_file", default="./vcsl_data/split_meta_pairs.json",
                        type=str, help="meta pair corresponding relation")

    args = parser.parse_args()

    hyper_params_search(args)

    logger.info("Finish hyper params tuning, evaluating...")

    eval_all(args)
