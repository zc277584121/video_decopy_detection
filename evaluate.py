import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='VCSL',
                        choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE", 'VCSL'],
                        help='Name of evaluation dataset.')
    parser.add_argument("--anno_file", default="vcsl_data/label_file_uuid_total.json", type=str, help="gt label file")
    parser.add_argument("--pair_group_file", default="vcsl_data/split_meta_pairs.json",
                        type=str, help="meta pair corresponding relation")
    parser.add_argument("--meta_info", default="vcsl_data/meta_info.json",
                        type=str, help="meta info with name and categories")
    parser.add_argument("--pred_file", type=str, required=True, help="result dir of video segment prediction")
    parser.add_argument("--split", type=str, default='test', help="store of pred data: train|val|test|all")
    parser.add_argument("--local_set_csv", type=str, help="save result csv file with query set, endswith '.csv' ")
    parser.add_argument("--pool_size", type=int, default=16, help="multiprocess pool size of evaluation")
    parser.add_argument("--ratio_pos_neg", type=float, default=1, help="ratio between positive and negative samples")

    parser.add_argument("--metric", type=str, required=True, choices=['f1', 'map'],
                        help="f1 metric or map metric, the input json file must have corresponding format")

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

        dataset = VCSL(datafolder='./vcsl_data', split=args.split)

    dataset.evaluate(args.pred_file, args.metric)
