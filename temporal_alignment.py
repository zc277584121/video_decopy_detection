import argparse
import os

import pandas as pd
from torch.utils.data import DataLoader
from loguru import logger
from itertools import product, islice

from utils import DataType, build_reader, build_writer
from vcsl.datasets import ItemDataset, filter_pair_list_by_np_dir
from vcsl.vta import build_vta_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--query_file", "-Q", type=str, help="data file")
    parser.add_argument("--reference_file", "-G", type=str, help="data file")
    parser.add_argument("--pair_file", type=str, help="data file")

    parser.add_argument("--dataset", type=str, help="dataset name, when using MPAA, please specify it, else not need.")

    # parser.add_argument("--input_store", type=str, help="store of input data: oss|local", default="oss")
    parser.add_argument("--input_root", type=str, help="root path of input data", default="")

    # parser.add_argument("--oss_config", type=str, default='~/ossutilconfig_copyright', help="url path")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="batch size")
    parser.add_argument("--data_workers", type=int, default=16, help="data workers")
    parser.add_argument("--request_workers", type=int, default=4, help="data workers")
    parser.add_argument("--output_root", type=str, help="output root")
    parser.add_argument("--output_store", type=str, help="store of output data: oss|local")

    # Hyper parameters or input model
    parser.add_argument("--alignment_method", type=str, default="DTW", help="DTW, DP, TN alignment method")

    parser.add_argument("--min_length",  type=int, default=5, help="minimum length of one segment")
    parser.add_argument("--sum_sim", type=float, default=10., help="minimum accumulated sim of one segment")
    parser.add_argument("--ave_sim", type=float, default=0.3, help="average sim of one segment")
    parser.add_argument("--min_sim", type=float, default=0.2, help="minimum average sim of one segment")

    parser.add_argument("--max_path", type=int, default=10, help="maximum number of paths to predict")
    parser.add_argument("--discontinue", type=int, default=3, help="max discontinue point in path")
    parser.add_argument("--max_iou", type=float, default=0.3, help="max iou to filter bboxes")

    parser.add_argument("--diagonal_thres", type=int, default=10, help="threshold for discarding a vertical/horizontal part of a segment for DP")

    parser.add_argument("--tn_top_K", type=int, default=5, help="top k nearest for TN")
    parser.add_argument("--tn_max_step", type=int, default=10, help="max step for TN")

    parser.add_argument("--angle_min", type=int, default=40, help="start angle for PM")
    parser.add_argument("--angle_max", type=int, default=50, help="end angle for PM")
    parser.add_argument("--interval", type=int, default=1, help="angle interval for PM")
    parser.add_argument("--peak_rate_thresh", type=float, default=0.85, help="peak rate thresh for PM")
    parser.add_argument("--peak_width_thresh", type=float, default=8, help="peak width thresh for PM")
    parser.add_argument("--accumulate_rate_thresh", type=float, default=0.4, help="accumulate rate thresh for PM")
    parser.add_argument("--matrix_h_thresh", type=float, default=100, help="matrix height thresh for PM")
    parser.add_argument("--reduce_size", type=int, default=4, help="reduce size for PM")

    parser.add_argument("--spd_model_path", type=str, help="SPD model path")
    parser.add_argument("--device", type=str, help="cpu or cuda:0 or others, only valid to SPD inference")
    parser.add_argument("--spd_conf_thres", type=float, default=0.5, help="bounding box conf filter for SPD inference")


    parser.add_argument("--params_file", type=str)

    parser.add_argument("--result_file", default="pred.json", type=str, help="result path")

    args = parser.parse_args()

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
    # config = dict()
    # if args.input_store == 'oss':
    #     config['oss_config'] = args.oss_config

    print('len(data_list) = ', len(data_list))
    dataset = ItemDataset(data_list,
                          store_type='local',
                          data_type=DataType.NUMPY.type_name,
                          root=args.input_root,
                          trans_key_func=lambda x: x + '.npy',
                          )

    logger.info(f"Data to run {len(dataset)}")

    loader = DataLoader(dataset, collate_fn=lambda x: x,
                        batch_size=args.batch_size,
                        num_workers=args.data_workers)

    model_config = dict()
    if args.alignment_method.startswith('DTW'):
        model_config = dict(
            discontinue=args.discontinue,
            min_sim=args.min_sim,
            min_length=args.min_length,
            max_iou=args.max_iou
        )
    elif args.alignment_method == 'TN':
        model_config = dict(
            tn_max_step=args.tn_max_step, tn_top_k=args.tn_top_K, max_path=args.max_path,
            min_sim=args.min_sim, min_length=args.min_length, max_iou=args.max_iou
        )
    elif args.alignment_method.startswith('DP'):
        model_config = dict(discontinue=args.discontinue,
                            min_sim=args.min_sim,
                            ave_sim=args.ave_sim,
                            min_length=args.min_length,
                            diagonal_thres=args.diagonal_thres)
    elif args.alignment_method.startswith('HV'):
        model_config = dict(min_sim=args.min_sim, iou_thresh=args.max_iou)
    elif args.alignment_method.startswith('SPD'):
        model_config = dict(model_path=args.spd_model_path,
                            conf_thresh=args.spd_conf_thres,
                            device=args.device)
    elif args.alignment_method == 'PM':
        model_config = dict(angle_min=args.angle_min,
                            angle_max=args.angle_max,
                            peak_rate_thresh=args.peak_rate_thresh,
                            peak_width_thresh=args.peak_width_thresh,
                            accumulate_rate_thresh=args.accumulate_rate_thresh,
                            matrix_h_thresh=args.matrix_h_thresh,
                            reduce_size=args.reduce_size
                            )
    elif args.alignment_method == 'TN+PM':
        model_config = dict(
            tn_max_step=args.tn_max_step, tn_top_k=args.tn_top_K, max_path=args.max_path,
            min_sim=args.min_sim, min_length=args.min_length, max_iou=args.max_iou,
            angle_min=args.angle_min,
            angle_max=args.angle_max,
            peak_rate_thresh=args.peak_rate_thresh,
            peak_width_thresh=args.peak_width_thresh,
            accumulate_rate_thresh=args.accumulate_rate_thresh,
            matrix_h_thresh=args.matrix_h_thresh,
            reduce_size=args.reduce_size
        )
    else:
        raise ValueError(f"Unknown VTA method: {args.alignment_method}")

    # override model config with param file
    if args.params_file:
        reader = build_reader('local', DataType.JSON.type_name)
        param_result = reader.read(args.params_file)
        best_params = param_result['best']
        logger.info("best param {}", best_params)
        model_config = best_params['param']

    model = build_vta_model(method=args.alignment_method, concurrency=args.request_workers, **model_config)

    total_result = dict()
    ii = 0
    for batch_data in islice(loader, 0, None):
        logger.info(f"{ii} / {len(loader)} data cnt: {len(batch_data)}, {batch_data[0][0]}")
        batch_result = model.forward_sim(batch_data)
        logger.info("result cnt: {}", len(batch_result))
        ii += 1
        for pair_id, result in batch_result:
            total_result[pair_id] = result

    output_store = 'local'
    if output_store == 'local' and not os.path.exists(args.output_root):
        os.makedirs(args.output_root, exist_ok=True)
    writer = build_writer(output_store, DataType.JSON.type_name)
    writer.write(os.path.join(args.output_root, args.result_file), total_result)
