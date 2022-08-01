import argparse
import os
import random
import torch
from tqdm import tqdm
import utils
from torch.utils.data import DataLoader
from datasets.generators import DatasetGenerator, DatasetGeneratorNpy
from model.students import FineGrainedStudent

random.seed(42)


def dns_index_features(model, data_list, args):
    generator = None
    if args.feature_path.endswith('.hdf5'):
        generator = DatasetGenerator(args.feature_path, data_list)
    elif os.path.isdir(args.feature_path):
        generator = DatasetGeneratorNpy(args.feature_path, data_list)
    loader = DataLoader(generator,  num_workers=args.workers, collate_fn=utils.collate_eval)
    # Extract features of the queries
    index_features = []
    print('\n> Extract features of the query videos')
    for video in tqdm(loader):
        video_features = video[0][0]
        video_id = video[2][0]
        if video_id:
            # print('video_features.shape = ', video_features.shape)
            features = model.index_video(video_features.to(args.device))
            if 'load_queries' in args and not args.load_queries: features = features.cpu()
            # all_db.add(video_id)
            index_features.append(features)
            # queries_ids.append(video_id)
    index_features = torch.cat(index_features, 0)

    return index_features




def calcu_similarity_matrix(dataset, args):
    # if args.feature_path.endswith('hdf5'):
    if args.similarity_type.lower() == 'dns':
        model = FineGrainedStudent(attention=args.dns_student_type == 'attention',
                                   binarization=args.dns_student_type == 'binarization',
                                   pretrained=True).to(args.device)
        queries_index_features = dns_index_features(model, dataset.get_queries(), args)
        targets_index_features = dns_index_features(model, dataset.get_database(), args)
        print('\n> Calculate query-target similarities')
        sims = model.calculate_video_similarity(queries_index_features, targets_index_features,visual_figure_name=None,
                                                visual_folder_path=args.visualization_dir).cpu().numpy()
        print('sims.shape=', sims.shape)
        return sims

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
    parser.add_argument('--visualization_dir', type=str, default='./sim_matrix_vis',
                        help='similarity matrix figure output dir, is None or '',  do not visualize.')
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
