import argparse
import os
import random
from itertools import islice

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.generators import DatasetGenerator, DatasetGeneratorNpy
from model.students import FineGrainedStudent
from utils import AsyncWriter, DataType
from utils import data_utils
from vcsl.datasets import PairDataset
from vcsl.vta import VideoSimMapModel

random.seed(42)


def load_features(data_list, args, dns_model: nn.Module=None):
    generator = None
    if args.feature_path.endswith('.hdf5'):
        generator = DatasetGenerator(args.feature_path, data_list)
    elif os.path.isdir(args.feature_path):
        generator = DatasetGeneratorNpy(args.feature_path, data_list)
    loader = DataLoader(generator, num_workers=args.workers, collate_fn=data_utils.collate_eval)
    for video in tqdm(loader):
        features = video[0][0]
        video_id = video[2][0]
        if video_id:
            if dns_model is not None:
                features = dns_model.index_video(features.to(args.device))
            yield video_id, features

def _dns_index_feature(dns_model, feature_np, device):
    features = torch.from_numpy(feature_np.astype(np.float32)).to(device)
    features = dns_model.index_video(features.to(device)).detach()
    return features

def calcu_similarity_matrix(dataset, args):
    if not os.path.exists(args.output_dir):
        print(f'\n > mkdir {args.output_dir}')
        os.mkdir(args.output_dir)

    writer_pool = AsyncWriter(pool_size=1,  # args.output_workers,#todo
                              store_type='local',
                              data_type=DataType.NUMPY.type_name,
                              )

    if args.similarity_type.lower() == 'dns':
        dns_model = FineGrainedStudent(attention=args.dns_student_type == 'attention',
                                   binarization=args.dns_student_type == 'binarization',
                                   pretrained=True).to(args.device)
        sim_map_model = None
    elif args.similarity_type.lower() in ["cos", "chamfer"]:
        dns_model = None
        sim_map_model = VideoSimMapModel(device=args.device)
    else:
        raise 'args.similarity_type must be in ["dns", "cos", "chamfer"]'
    targets_generator = load_features(dataset.get_database(), args, dns_model=dns_model)
    queries_generator = load_features(dataset.get_queries(), args, dns_model=dns_model)

    pair_dataset = PairDataset(query_list=None,
                          gallery_list=None,
                          pair_list=dataset.get_pairs(args.pair_file),
                          file_dict=dataset.get_files_dict(),
                          root=args.feature_path,
                          store_type='local',
                          trans_key_func=lambda x: x + ".npy",
                          data_type="numpy",
                          )
    pair_loader = DataLoader(pair_dataset, collate_fn=lambda x: x,
                        batch_size=4,
                        num_workers=0 if args.similarity_type.lower() == 'dns' else 4)#args.workers) #todo


    if args.pair_file is None: # query_reference
        batch_sz = 2048 if 'batch_sz_sim' not in args else args.batch_sz_sim
        print('\n> Extract features of the query videos')
        queries_index_infos = []
        for queries_index_info in queries_generator:
            queries_index_infos.append(queries_index_info)
        print('\n> Calculate query-target similarities')


        # exist_files = os.listdir('./sim_matrix_npy/mpaa-dns_backbone-qd_pair-dns_sim')

        for targets_index_info in targets_generator:
            for queries_index_info in queries_index_infos:
                batch_matrix_list = []
                query_id = queries_index_info[0]
                target_id = targets_index_info[0]

                # npy_file = f'{query_id}-{target_id}.npy'
                # if npy_file in exist_files:
                #     # print('already exist, skip...')
                #     continue

                for batch_idx in range(
                        targets_index_info[1].shape[0] // batch_sz + 1):  # we can reduce batch_sz to avoid OOM
                    targets_feature_batch = targets_index_info[1][batch_idx * batch_sz: (batch_idx + 1) * batch_sz]
                    if targets_feature_batch.shape[0] >= 4:
                        query_feature = queries_index_info[1]
                        if args.similarity_type.lower() == 'dns':
                            batch_sim_matrix = dns_model.calculate_similarity_matrix(query_feature, targets_feature_batch).detach() #batch size is 0
                        elif args.similarity_type.lower() in ["cos", "chamfer"]:
                            _, _, batch_sim_matrix = sim_map_model.forward([(query_id, target_id, query_feature, targets_feature_batch)],
                                                                           normalize_input=args.with_l2_norm, similarity_type=args.similarity_type)[0] #batch size is 1 so use `[]`
                            batch_sim_matrix = torch.Tensor(batch_sim_matrix)
                        else:
                            raise 'args.similarity_type must be in ["dns", "cos", "chamfer"]'
                        batch_matrix_list.append(batch_sim_matrix)
                sim_matrix = torch.concat(batch_matrix_list, dim=1)
                sim_matrix = sim_matrix.cpu().numpy()
                key = os.path.join(args.output_dir, f"{query_id}-{target_id}.npy")
                writer_pool.consume((key, sim_matrix))
        writer_pool.stop()
    else:  # pair
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

        for batch_data in tqdm(pair_loader):
            if args.similarity_type.lower() == 'dns':
                query_id_batch = [b[0] for b in batch_data if b[2] is not None]
                target_id_batch = [b[1] for b in batch_data if b[3] is not None]
                query_batch = [_dns_index_feature(dns_model, b[2], args.device) for b in batch_data if b[2] is not None]
                target_batch = [_dns_index_feature(dns_model, b[3], args.device) for b in batch_data if b[3] is not None]
                batch_result = []
                for query_id, target_id, query, target in zip(query_id_batch, target_id_batch, query_batch, target_batch):
                    sim_matrix = dns_model.calculate_similarity_matrix(query, target).detach().cpu().numpy() #batch size is 0
                    batch_result.append([query_id, target_id, sim_matrix])
            elif args.similarity_type.lower() in ["cos", "chamfer"]:
                batch_result = sim_map_model.forward(batch_data, normalize_input=args.with_l2_norm, similarity_type=args.similarity_type) #batch size > 0
            else:
                raise 'args.similarity_type must be in ["dns", "cos", "chamfer"]'

            for q_id, r_id, result in batch_result:
                key = os.path.join(args.output_dir, f"{q_id}-{r_id}.npy")
                writer_pool.consume((key, result))
        writer_pool.stop()


if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='feature extracting', formatter_class=formatter)
    parser.add_argument('--dataset', type=str, default='VCSL',
                        choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE", 'VCSL', 'MPAA', 'MUSCLE_VCD'],
                        help='Name of evaluation dataset.')
    parser.add_argument('--video_root', type=str,
                        help='video root path, not necessarily required.')
    parser.add_argument('--feature_path', type=str, required=True,
                        help='feature path or dir, it can be a hdf5 file or a dir which contains npy files')
    parser.add_argument("--similarity_type", default='cos',
                        choices=["cos", "chamfer", "DnS"],
                        type=str, help="cos or chamfer")
    parser.add_argument("--dns_student_type", default='attention',
                        choices=["attention", "binarization"],
                        type=str, help="valid when similarity_type is DnS")
    parser.add_argument('--pair_file', type=str, default=None,
                        help='if pair_file is not None, it will use pairs in pair_file to calculate similarity matrix, instead of query-database wise pairs')
    parser.add_argument('--output_dir', type=str, default='./sim_matrix_npy',
                        help='similarity matrix output dir.')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of workers used for video loading.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='ID of the GPU.')
    parser.add_argument('--with_l2_norm', type=bool, default=True,
                        help='If cos, use l2_norm before similarity calculating')
    parser.add_argument('--batch_sz_sim', type=int, default=2048,
                        help='To avoid OOM, you can reduce this value when calculating dns similarity matrix.')
    args = parser.parse_args()

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

        dataset = VCSL(datafolder='./vcsl_data', split='val')

    elif 'MPAA' in args.dataset:
        from datasets import MPAA
        dataset = MPAA(video_root=args.video_root)

    elif 'MUSCLE_VCD' in args.dataset:
        from datasets import MUSCLE_VCD
        dataset = MUSCLE_VCD(video_root='./muscle_vcd')

    calcu_similarity_matrix(dataset, args)
