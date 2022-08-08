import argparse
import json
import os
from functools import partial
from multiprocessing import Pool
from typing import Dict

# import h5py
import numpy as np
import pandas as pd

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


def hyper_params_search(args):
    pairs, files_dict, query, reference = None, None, None, None
    if args.pair_file:
        df = pd.read_csv(args.pair_file)
        pairs = df[['query_id', 'reference_id']].values.tolist()

        pairs = filter_pair_list_by_np_dir(pairs, args.input_root)


        data_list = [(f"{p[0]}-{p[1]}", f"{p[0]}-{p[1]}") for p in pairs]
    else:
        query = pd.read_csv(args.query_file)
        query = query[['uuid']].values.tolist()

        reference = pd.read_csv(args.reference_file)
        reference = reference[['uuid']].values.tolist()

        pairs = product(query, reference)
        data_list = [(f"{p[0]}-{p[1]}", f"{p[0]}-{p[1]}") for p in pairs]

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

    elif args.alignment_method.startswith('TN'):
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
        if output_store == 'local':
            os.makedirs(os.path.join(args.output_root, sub_dir), exist_ok=True)

        for batch_data in islice(loader, 0, None):
            logger.info("data cnt: {}, {}", len(batch_data), batch_data[0][0])
            batch_result = model.forward_sim(batch_data)
            logger.info("result cnt: {}", len(batch_result))

            for pair_id, result in batch_result:
                key = os.path.join(args.output_root, sub_dir, f"{pair_id}.json")
                writer_pool.consume((key, result))

    writer_pool.stop()


def eval_all(args):
    gt = json.load(open(args.anno_file))
    key_list = [key for key in gt]
    root_dir = os.path.dirname(args.anno_file)
    split_file = os.path.join(root_dir, f"pair_file_val.csv")
    df = pd.read_csv(split_file)
    # split_pairs = set([f"{q}-{r}" for q, r in zip(df.query_id.values, df.reference_id.values)])

    split_pairs = df[['query_id', 'reference_id']].values.tolist()
    split_pairs = filter_pair_list_by_np_dir(split_pairs, args.input_root)
    split_pairs = set([f"{q_r[0]}-{q_r[1]}" for q_r in split_pairs])


    logger.info("Val set contains pairs {}", len(split_pairs))

    key_list = [key for key in key_list if key in split_pairs]
    print('len(gt intersection split file) =', len(key_list))

    pair_group_dict = json.load(open(args.pair_group_file))['val']

    print('len(query file in pair_group_file)=' , len(pair_group_dict))

    output_store = 'local'
    reader = build_reader(output_store, "bytes")

    process_pool = Pool(args.data_workers)

    final_results = {'results': [], 'best': None}
    best, best_idx = dict(result=dict(f1=-1)), -1

    hyper_params = json.loads(reader.read(os.path.join(args.output_root, 'hyper_params.json')))
    for idx, param in enumerate(islice(hyper_params, 0, None)):
        data_root = os.path.join(args.output_root, str(idx))

        read_func = partial(gen_input, reader=reader, root=data_root, gt=gt)
        eval_list = process_pool.map(read_func, key_list)

        logger.info(f"finish loading files, start evaluation...")

        result_list = process_pool.map(run_eval, eval_list)
        result_dict = {i['name']: i for i in result_list}
        r, p, cnt = evaluate_macro(result_dict, pair_group_dict)
        f1 = 2 * r * p / (r + p)
        logger.info(f"Hyper params {idx}: {param}")
        logger.info(
            f"query set cnt {cnt}, "
            f"query macro-Recall: {r:.2%}, "
            f"query macro-Precision: {p:.2%}, "
            f"F1: {f1:.2%}")

        result = dict(param=param, result=dict(r=r, p=p, f1=f1))
        best, best_idx = (result, idx) \
            if f1 > best['result']['f1'] else (best, best_idx)
        final_results['results'].append(result)
        logger.info('')

    final_results['best'] = best
    final_results['best_idx'] = best_idx
    logger.info(f'best params {best_idx}: {best["param"]}')
    logger.info(f'best result: {best["result"]}')

    writer = build_writer(output_store, DataType.BYTES.type_name)
    d = json.dumps(final_results, indent=2, ensure_ascii=False)
    writer.write(os.path.join(args.output_root, 'result.json'), d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--query_file", "-Q", type=str, help="data file")
    parser.add_argument("--reference_file", "-G", type=str, help="data file")
    parser.add_argument("--pair_file", type=str, help="data file")

    # parser.add_argument("--input_store", type=str, help="store of input data: oss|local", default="oss")
    parser.add_argument("--input_root", type=str, help="root path of input data", default="")

    # parser.add_argument("--oss_config", type=str, default='~/ossutilconfig_copyright', help="url path")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="batch size")
    parser.add_argument("--data_workers", type=int, default=32, help="data workers")
    parser.add_argument("--request_workers", type=int, default=4, help="data workers")
    parser.add_argument("--output_workers", type=int, default=8, help="oss upload workers")
    parser.add_argument("--output_root", type=str, help="output root")
    parser.add_argument("--output_store", type=str, help="store of input data: oss|local")

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

    parser.add_argument("--result_prefix", type=str, help="result path")

    parser.add_argument("--anno_file", default="./vcsl_data/label_file_uuid_total.json", type=str, help="gt label file")
    parser.add_argument("--pair_group_file", default="./vcsl_data/split_meta_pairs.json",
                        type=str, help="meta pair corresponding relation")

    args = parser.parse_args()

    hyper_params_search(args)

    logger.info("Finish hyper params tuning, evaluating...")

    eval_all(args)


