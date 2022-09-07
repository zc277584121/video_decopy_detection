import argparse
import random
from pathlib import Path

import timm

from datasets import load_video, load_video_by_video_decode
from tqdm import tqdm
import h5py
import torch
from model.feature_extractor import DnSR50FeatureExtractor, TimmFeatureExtractor, IscFeatureExtractor, DinoFeatureExtractor, MAEFeatureExtractor
import numpy as np

random.seed(42)


def get_one_video_hdf5_features(video_id, video_path, feature_extractor, max_interval=256, device='cuda:0',
                                output_name='./features/features.hdf5', crop_resize=256):
    """
    To avoid OOM, `max_interval` is used for split video tensor to several clip chunks, and concat them after feature extracting.
    """
    video_numpy = load_video(video_path, crop_resize=crop_resize)
    if (video_numpy.shape == (0,)):
        print('video_numpy.shape is zero, try to use video decode to extract feature...')
        video_numpy = load_video_by_video_decode(video_path, crop_resize=crop_resize)


    chunk_num = video_numpy.shape[0] // max_interval + 1
    print('chunk_num = ', chunk_num)
    chunk_list = np.array_split(video_numpy, chunk_num, axis=0)
    feature_list = []
    for chunk in chunk_list:
        print('chunk.shape=', chunk.shape)
        if feature_extractor.__class__.__name__ == 'DnSR50FeatureExtractor':
            chunk = torch.from_numpy(chunk).to(device)
        video_features = feature_extractor(chunk)#.to(device)
        video_features = video_features.detach().cpu().numpy()
        feature_list.append(video_features)
    video_features = np.concatenate(feature_list, axis=0)


    with h5py.File(output_name, 'a') as hdf5_file:
    # with h5py.File(output_name, 'w') as hdf5_file:
        hdf5_features = hdf5_file.create_dataset(video_id, data=video_features, dtype="f", compression='gzip',
                                                 compression_opts=9)
    return video_features


def get_hdf5_features(dataset, args):
    all_data_file_list = dataset.all_data_file_list
    all_data_id_list = dataset.all_data_id_list
    feature_extractor = None
    crop_resize = 256
    if args.feature_backbone == 'DnS_R50':
        crop_resize = 256
        feature_extractor = DnSR50FeatureExtractor(dims=512).to(args.device).eval()

    elif args.feature_backbone in timm.list_models():
        crop_resize = None
        feature_extractor = TimmFeatureExtractor(model_name=args.feature_backbone, device=args.device)
    elif args.feature_backbone == 'ISC':
        crop_resize = None
        feature_extractor = IscFeatureExtractor(device=args.device)
    elif args.feature_backbone == 'DINO':
        crop_resize = None
        feature_extractor = DinoFeatureExtractor(device=args.device)
    elif args.feature_backbone == 'MAE':
        crop_resize = None
        feature_extractor = MAEFeatureExtractor(device=args.device, chkpt_dir=args.chkpt_dir, arch=args.mae_arch)

    all_video_path_lists = []
    i = 0
    with open('feature_extact_failed.txt', 'w') as f:
        for video_id, file_name in tqdm(zip(all_data_id_list, all_data_file_list)):
            i += 1
            video_path = str(file_name)
            # if video_path not in dataset_split_paths: #todo
            #     continue
            print('video_id =', video_id)
            print('video_path =', video_path)
            print(f'{i} / {len(all_data_id_list)}')
            try:
                feature = get_one_video_hdf5_features(video_id, video_path, feature_extractor, max_interval=256,
                                                      device=args.device, output_name=args.output_name,
                                                      crop_resize=crop_resize)
                print('feature.shape =', feature.shape)
                # pass
            except Exception as e:
                print('error!!!\n')
                print(e)
                f.write(video_path + '\n')
                continue
            print('\n')


if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='feature extracting', formatter_class=formatter)
    parser.add_argument('--dataset', type=str, default='VCSL',
                        choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE", 'VCSL', 'MPAA', 'MUSCLE_VCD'],
                        help='Name of evaluation dataset.')
    parser.add_argument('--feature_backbone', type=str, default='DnS_R50',
                        choices=["DnS_R50", 'ISC', 'DINO', 'MAE'] + timm.list_models(),
                        help='backbone to extract feature')
    parser.add_argument('--output_type', type=str, default='hdf5', choices=["hdf5", "npy"],
                        help='output feature type.')
    parser.add_argument('--output_name', type=str, required=True,
                        help='video output dir or file path.')
    parser.add_argument('--video_root', type=str, required=True,
                        help='video root dir.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device,CPU or GPU.')
    parser.add_argument('--chkpt_dir', type=str, default='',
                        help='Checkpoint directory for MAE.')
    parser.add_argument('--mae_arch', type=str, default='mae_vit_large_patch16',
                        help='Arch for ViT.')
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

        dataset = VCSL(datafolder='./vcsl_data', split='val', video_root=args.video_root)

    elif 'MPAA' in args.dataset:
        from datasets import MPAA
        dataset = MPAA(video_root=args.video_root)

    elif 'MUSCLE_VCD' in args.dataset:
        from datasets import MUSCLE_VCD
        dataset = MUSCLE_VCD(video_root=args.video_root)


    if args.output_type == 'hdf5':
        get_hdf5_features(dataset, args)
    # elif args.output_type == 'npy': todo:
    #     get_npy_features(args)
