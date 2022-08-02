import argparse
import os
import random
import torch
from tqdm import tqdm
# import utils
from torch.utils.data import DataLoader
from datasets.generators import DatasetGenerator, DatasetGeneratorNpy
from model.students import FineGrainedStudent
from utils import AsyncWriter, DataType
from utils import data_utils

random.seed(42)


def dns_index_features(model, data_list, args):
    generator = None
    if args.feature_path.endswith('.hdf5'):
        generator = DatasetGenerator(args.feature_path, data_list)
    elif os.path.isdir(args.feature_path):
        generator = DatasetGeneratorNpy(args.feature_path, data_list)
    loader = DataLoader(generator, num_workers=args.workers, collate_fn=data_utils.collate_eval)
    for video in tqdm(loader):
        video_features = video[0][0]
        video_id = video[2][0]
        if video_id:
            features = model.index_video(video_features.to(args.device))
            yield video_id, features


def calcu_similarity_matrix(dataset, args):
    if not os.path.exists(args.output_dir):
        print(f'\n > mkdir {args.output_dir}')
        os.mkdir(args.output_dir)

    writer_pool = AsyncWriter(pool_size=4,  # args.output_workers,#todo
                              store_type='local',
                              data_type=DataType.NUMPY.type_name,
                              )

    if args.similarity_type.lower() == 'dns':
        model = FineGrainedStudent(attention=args.dns_student_type == 'attention',
                                   binarization=args.dns_student_type == 'binarization',
                                   pretrained=True).to(args.device)
        batch_sz = 2048 if 'batch_sz_sim' not in args else args.batch_sz_sim
        targets_index_generator = dns_index_features(model, dataset.get_database(), args)

        print('\n> Extract features of the query videos')
        queries_index_generator = dns_index_features(model, dataset.get_queries(), args)
        queries_index_infos = []
        for queries_index_info in queries_index_generator:
            queries_index_infos.append(queries_index_info)
        print('\n> Calculate query-target similarities')
        for targets_index_info in targets_index_generator:
            for queries_index_info in queries_index_infos:
                batch_matrix_list = []
                query_id = queries_index_info[0]
                target_id = targets_index_info[0]

                for batch_idx in range(
                        targets_index_info[1].shape[0] // batch_sz + 1):  # we can reduce batch_sz to avoid OOM
                    targets_feature_batch = targets_index_info[1][batch_idx * batch_sz: (batch_idx + 1) * batch_sz]
                    if targets_feature_batch.shape[0] >= 4:
                        query_feature = queries_index_info[1]
                        batch_sim_matrix = model.calculate_similarity_matrix(query_feature, targets_feature_batch)
                        batch_sim_matrix = batch_sim_matrix.detach()[0]
                        batch_matrix_list.append(batch_sim_matrix)
                sim_matrix = torch.concat(batch_matrix_list, dim=1)
                sim_matrix = sim_matrix.cpu().numpy()
                key = os.path.join(args.output_dir, f"{target_id}-{query_id}.npy")
                writer_pool.consume((key, sim_matrix))
        writer_pool.stop()


if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='feature extracting', formatter_class=formatter)
    parser.add_argument('--dataset', type=str, default='VCSL',
                        choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE", 'VCSL'],
                        help='Name of evaluation dataset.')
    parser.add_argument('--feature_path', type=str, required=True,
                        help='feature path or dir, it can be a hdf5 file or a dir which contains npy files')
    parser.add_argument("--similarity_type", default='cos',
                        choices=["cos", "chamfer", "DnS"],
                        type=str, help="cos or chamfer")
    parser.add_argument("--dns_student_type", default='attention',
                        choices=["attention", "binarization"],
                        type=str, help="valid when similarity_type is DnS")
    parser.add_argument('--output_dir', type=str, default='./sim_matrix_npy',
                        help='similarity matrix output dir.')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of workers used for video loading.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='ID of the GPU.')
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

    calcu_similarity_matrix(dataset, args)
