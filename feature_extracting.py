import argparse
import random
from pathlib import Path
from datasets import load_video
from tqdm import tqdm
import h5py
import torch
from model.feature_extractor import FeatureExtractor as DnSR50FeatureExtractor
import numpy as np

random.seed(42)


def get_one_video_hdf5_features(video_path, feature_extractor, max_interval=256, device='cuda:0',
                                output_name='./features/features.hdf5'):
    """
    To avoid OOM, `max_interval` is used for split video tensor to several clip chunks, and concat them after feature extracting.
    """
    video_numpy = load_video(video_path)
    chunk_num = video_numpy.shape[0] // max_interval + 1
    print('chunk_num = ', chunk_num)
    chunk_list = np.array_split(video_numpy, chunk_num, axis=0)

    feature_list = []
    for chunk in chunk_list:
        video_tensor = torch.from_numpy(chunk).to(device)
        video_features = feature_extractor(video_tensor).to(device)
        video_features = video_features.detach().cpu().numpy()
        feature_list.append(video_features)
    video_features = np.concatenate(feature_list, axis=0)
    video_id = video_path.split('/')[-1].split('.')[0].split('-')[0]
    print('video_id = ', video_id)
    # with h5py.File(output_name, 'a') as hdf5_file:
    with h5py.File(output_name, 'w') as hdf5_file:
        hdf5_features = hdf5_file.create_dataset(video_id, data=video_features, dtype="f", compression='gzip',
                                                 compression_opts=9)
    return video_features


def get_hdf5_features(dataset, args):
    root_dir = dataset.video_root
    feature_extractor = None
    if args.feature_backbone == 'DnS_R50':
        feature_extractor = DnSR50FeatureExtractor(dims=512).to(args.device).eval()
    # elif ...    #todo

    all_video_path_lists = []
    with open('feature_extact_failed.txt', 'w') as f:
        for file_name in tqdm(Path(root_dir).iterdir()):
            video_path = str(file_name)
            # if video_path not in dataset_split_paths: #todo
            #     continue
            print('video_path =', video_path)
            try:
                feature = get_one_video_hdf5_features(video_path, feature_extractor, max_interval=256,
                                                      device=args.device, output_name=args.output_name)
                print('feature.shape =', feature.shape)
            except:
                print('error!!!\n')
                f.write(video_path + '\n')
                continue
            print('\n')


if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='feature extracting', formatter_class=formatter)
    parser.add_argument('--dataset', type=str, default='VCSL',
                        choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE", 'VCSL'],
                        help='Name of evaluation dataset.')
    parser.add_argument('--feature_backbone', type=str, default='DnS_R50',
                        choices=["DnS_R50", ],  # todo: more kinds of feature
                        help='backbone to extract feature')
    parser.add_argument('--output_type', type=str, default='hdf5', choices=["hdf5", "npy"],
                        help='output feature type.')
    parser.add_argument('--output_name', type=str, required=True,
                        help='video output dir or file path.')
    parser.add_argument('--video_root', type=str, required=True,
                        help='video root dir.')
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

        dataset = VCSL(datafolder='./vcsl_data', split='all', video_root=args.video_root)

    if args.output_type == 'hdf5':
        get_hdf5_features(dataset, args)
    # elif args.output_type == 'npy': todo:
    #     get_npy_features(args)
