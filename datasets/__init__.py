import json
import os
from pprint import pprint
from typing import List

import h5py
import numpy as np
import pickle as pk
import cv2

from collections import OrderedDict, defaultdict

import pandas as pd
from sklearn.metrics import average_precision_score
from multiprocessing import Pool

# from analyse_result import get_mpaa_video_file_path
from utils import build_reader
from vcsl.metric import evaluate_micro, evaluate_macro, precision_recall
import re


def match_pattern(string, pattern):
    if re.search(pattern, string) is None:
        return False
    else:
        return True


def get_mpaa_video_file_path(pair_str, video_root='/home/zhangchen/zhangchen_workspace/dataset/frank/MPAA',
                             return_all_path=True):
    file_ext_set = {'wmv', 'flv', 'out', 'MPG', 'vog', 'mpg', 'mov', 'VOB', 'avi', 'sh', 'txt', 'vob',
                    'rmvb', 'mpeg', 'mp4'}
    split_plan_num = 0
    for file_ext in file_ext_set:
        if len(pair_str.split(f'.{file_ext}-')) == 2:
            query_id = pair_str.split(f'.{file_ext}-')[0] + f'.{file_ext}'
            ref_id = pair_str.split(f'.{file_ext}-')[1]
            split_plan_num += 1
    if split_plan_num != 1:
        raise RuntimeError(f'{pair_str} can not split well.')
    # print(pair_str)
    # print(query_id, ref_id)
    # print('')
    if return_all_path:
        query_path = os.path.join(video_root, 'all', query_id)
        ref_path = os.path.join(video_root, 'master', ref_id)
    else:
        query_path = query_id
        ref_path = ref_id
    return query_path, ref_path


def run_eval(input_dict):
    gt_box = np.array(input_dict["gt"])
    pred_box = np.array(input_dict["pred"])
    result_dict = precision_recall(pred_box, gt_box)
    result_dict["name"] = input_dict["name"]
    return result_dict


def resize_frame(frame, desired_size):
    min_size = np.min(frame.shape[:2])
    ratio = desired_size / min_size
    frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return frame


def center_crop(frame, desired_size):
    old_size = frame.shape[:2]
    top = int(np.maximum(0, (old_size[0] - desired_size) / 2))
    left = int(np.maximum(0, (old_size[1] - desired_size) / 2))
    return frame[top: top + desired_size, left: left + desired_size, :]


def load_video(video, all_frames=False, crop_resize=256):
    cv2.setNumThreads(3)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('fps = ', fps)
    print('frame_count = ', frame_count)
    duration = frame_count / fps / 60
    print('duration = ', duration)

    if fps > 144 or fps is None:
        print('Abnormal fps')
        fps = 25
    frames = []
    count = 0
    print('fps = ', fps)

    # while cap.isOpened():
    #     ret = cap.grab()
    #     if int(count % round(fps)) == 0 or all_frames:
    #         print(count)
    #         ret, frame = cap.retrieve()
    #         if isinstance(frame, np.ndarray):
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             frames.append(center_crop(resize_frame(frame, 256), 256))
    #         else:
    #             break
    #     count += 1
    # # print('count = ', count)
    # cap.release()
    # return np.array(frames)

    while cap.isOpened():
        # ret = cap.grab()
        ret, frame = cap.read()
        # print('ret = ', ret)
        if ret:
            if int(count % round(fps)) == 0 or all_frames:
                # print(f'{count}/{int(frame_count)}')
                # ret, frame = cap.retrieve()
                if isinstance(frame, np.ndarray):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if crop_resize is not None:
                        crop_frame = center_crop(resize_frame(frame, crop_resize), crop_resize)
                    else:
                        crop_frame = resize_frame(frame, 256)
                    frames.append(crop_frame)
                else:
                    break
        else:
            break
        if count > frame_count:
            print('count > frame_count, break')
            break
        count += 1
    # print('count = ', count)
    cap.release()
    return np.array(frames)


def load_video_by_video_decode(video_path, crop_resize=256):
    from towhee import ops
    decode_op = ops.video_decode.ffmpeg(start_time=None, end_time=None, sample_type='time_step_sample',
                                        args={'time_step': 1})
    frame_list = []
    for frame in decode_op(video_path):
        # print(frame)
        if crop_resize is not None:
            crop_frame = center_crop(resize_frame(frame, crop_resize), crop_resize)
        else:
            crop_frame = resize_frame(frame, 256)
        frame_list.append(crop_frame)
    video_frames_np = np.array(frame_list)
    return video_frames_np


class CC_WEB_VIDEO(object):

    def __init__(self):
        with open('datasets/cc_web_video.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.name = 'CC_WEB_VIDEO'
        self.index = dataset['index']
        self.queries = list(dataset['queries'])
        self.database = sorted(list(map(str, self.index.keys())))
        self.ground_truth = dataset['ground_truth']
        self.excluded = dataset['excluded']

    def get_queries(self):
        return self.queries

    def get_database(self):
        return self.database

    def calculate_mAP(self, similarities, all_db, all_videos=False, clean=False, positive_labels='ESLMV'):
        mAP = 0.0
        for query_set, labels in enumerate(self.ground_truth):
            query_id = self.queries[query_set]
            i, ri, s = 0.0, 0.0, 0.0
            if query_id in similarities:
                res = similarities[query_id]
                if isinstance(res, (np.ndarray, np.generic)):
                    res = {v: s for v, s in zip(self.database, res) if v in all_db}
                for video_id in sorted(res.keys(), key=lambda x: res[x], reverse=True):
                    if video_id not in self.index: continue
                    video = self.index[video_id]
                    if (all_videos or video in labels) and (not clean or video not in self.excluded[query_set]):
                        ri += 1
                        if video in labels and labels[video] in positive_labels:
                            i += 1.0
                            s += i / ri
                positives = np.sum([1.0 for k, v in labels.items() if
                                    v in positive_labels and (not clean or k not in self.excluded[query_set])])
                mAP += s / positives
        return mAP / len(set(self.queries).intersection(similarities.keys()))

    def evaluate(self, similarities, all_db=None, verbose=True):
        if all_db is None:
            all_db = set(self.database)

        if verbose:
            print('=' * 5, 'CC_WEB_VIDEO Dataset', '=' * 5)
            not_found = len(set(self.queries) - similarities.keys())
            if not_found > 0:
                print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
            print('Queries: {} videos'.format(len(similarities)))
            print('Database: {} videos'.format(len(all_db)))

        mAP = self.calculate_mAP(similarities, all_db, all_videos=False, clean=False)
        mAP_star = self.calculate_mAP(similarities, all_db, all_videos=True, clean=False)
        if verbose:
            print('-' * 25)
            print('All dataset')
            print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}\n'.format(mAP, mAP_star))

        mAP_c = self.calculate_mAP(similarities, all_db, all_videos=False, clean=True)
        mAP_c_star = self.calculate_mAP(similarities, all_db, all_videos=True, clean=True)
        if verbose:
            print('Clean dataset')
            print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}'.format(mAP_c, mAP_c_star))
        return {'mAP': mAP, 'mAP_star': mAP_star, 'mAP_c': mAP_c, 'mAP_c_star': mAP_c_star}


class FIVR(object):

    def __init__(self, version='200k', audio=False):
        self.version = version
        self.audio = audio
        with open('datasets/fivr.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.name = 'FIVR'
        self.annotation = dataset['annotation']
        if not self.audio:
            self.queries = sorted(list(dataset[self.version]['queries']))
        else:
            self.queries = sorted([q for q in dataset[self.version]['queries'] if 'DA' in self.annotation[q]])
        self.database = sorted(list(dataset[self.version]['database']))

    def get_queries(self):
        return self.queries

    def get_database(self):
        return self.database

    def calculate_mAP(self, query, res, all_db, relevant_labels):
        gt_sets = self.annotation[query]
        query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))
        query_gt = query_gt.intersection(all_db)

        i, ri, s = 0.0, 0, 0.0
        for video in sorted(res.keys(), key=lambda x: res[x], reverse=True):
            if video != query and video in all_db:
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri
        return s / len(query_gt)

    def evaluate(self, similarities, all_db=None, verbose=True):
        if all_db is None:
            all_db = set(self.database)
        else:
            all_db = set(self.database).intersection(all_db)

        if not self.audio:
            DSVR, CSVR, ISVR = [], [], []
            for query, res in similarities.items():
                if query in self.queries:
                    if isinstance(res, (np.ndarray, np.generic)):
                        res = {v: s for v, s in zip(self.database, res) if v in all_db}
                    DSVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS']))
                    CSVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS', 'CS']))
                    ISVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS', 'CS', 'IS']))
            if verbose:
                print('=' * 5, 'FIVR-{} Dataset'.format(self.version.upper()), '=' * 5)
                not_found = len(set(self.queries) - similarities.keys())
                if not_found > 0:
                    print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))

                print('Queries: {} videos'.format(len(similarities)))
                print('Database: {} videos'.format(len(all_db)))

                print('-' * 16)
                print('DSVR mAP: {:.4f}'.format(np.mean(DSVR)))
                print('CSVR mAP: {:.4f}'.format(np.mean(CSVR)))
                print('ISVR mAP: {:.4f}'.format(np.mean(ISVR)))
            return {'DSVR': np.mean(DSVR), 'CSVR': np.mean(CSVR), 'ISVR': np.mean(ISVR)}
        else:
            DAVR = []
            for query, res in similarities.items():
                if query in self.queries:
                    if isinstance(res, (np.ndarray, np.generic)):
                        res = {v: s for v, s in zip(self.database, res) if v in all_db}
                    DAVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['DA']))
            if verbose:
                print('=' * 5, 'FIVR-{} Dataset'.format(self.version.upper()), '=' * 5)
                not_found = len(set(self.queries) - similarities.keys())
                if not_found > 0:
                    print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))

                print('Queries: {} videos'.format(len(similarities)))
                print('Database: {} videos'.format(len(all_db)))

                print('-' * 16)
                print('DAVR mAP: {:.4f}'.format(np.mean(DAVR)))
            return {'DAVR': np.mean(DAVR)}


class EVVE(object):

    def __init__(self):
        with open('datasets/evve.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.name = 'EVVE'
        self.events = dataset['annotation']
        self.queries = sorted(list(dataset['queries']))
        self.database = sorted(list(dataset['database']))
        self.query_to_event = {qname: evname
                               for evname, (queries, _, _) in self.events.items()
                               for qname in queries}

    def get_queries(self):
        return self.queries

    def get_database(self):
        return self.database

    def score_ap_from_ranks_1(self, ranks, nres):
        """ Compute the average precision of one search.
        ranks = ordered list of ranks of true positives (best rank = 0)
        nres  = total number of positives in dataset
        """
        if nres == 0 or ranks == []:
            return 0.0

        ap = 0.0

        # accumulate trapezoids in PR-plot. All have an x-size of:
        recall_step = 1.0 / nres

        for ntp, rank in enumerate(ranks):
            # ntp = nb of true positives so far
            # rank = nb of retrieved items so far

            # y-size on left side of trapezoid:
            if rank == 0:
                precision_0 = 1.0
            else:
                precision_0 = ntp / float(rank)
            # y-size on right side of trapezoid:
            precision_1 = (ntp + 1) / float(rank + 1)
            ap += (precision_1 + precision_0) * recall_step / 2.0
        return ap

    def evaluate(self, similarities, all_db=None, verbose=True):
        results = {e: [] for e in self.events}
        if all_db is None:
            all_db = set(self.database).union(set(self.queries))

        not_found = 0
        for query in self.queries:
            if query not in similarities:
                not_found += 1
            else:
                res = similarities[query]
                if isinstance(res, (np.ndarray, np.generic)):
                    res = {v: s for v, s in zip(self.database, res) if v in all_db}
                evname = self.query_to_event[query]
                _, pos, null = self.events[evname]
                if all_db:
                    pos = pos.intersection(all_db)
                pos_ranks = []

                ri, n_ext = 0.0, 0.0
                for ri, dbname in enumerate(sorted(res.keys(), key=lambda x: res[x], reverse=True)):
                    if dbname in pos:
                        pos_ranks.append(ri - n_ext)
                    if dbname not in all_db:
                        n_ext += 1

                ap = self.score_ap_from_ranks_1(pos_ranks, len(pos))
                results[evname].append(ap)
        if verbose:
            print('=' * 18, 'EVVE Dataset', '=' * 18)
            if not_found > 0:
                print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
            print('Queries: {} videos'.format(len(similarities)))
            print('Database: {} videos\n'.format(len(all_db - set(self.queries))))
            print('-' * 50)
        ap, mAP = [], []
        for evname in sorted(self.events):
            queries, _, _ = self.events[evname]
            nq = len(queries.intersection(all_db))
            ap.extend(results[evname])
            mAP.append(np.sum(results[evname]) / nq)
            if verbose:
                print('{0: <36} '.format(evname), 'mAP = {:.4f}'.format(np.sum(results[evname]) / nq))

        if verbose:
            print('=' * 50)
            print('overall mAP = {:.4f}'.format(np.mean(ap)))
        return {'mAP': np.mean(ap)}


class SVD(object):

    def __init__(self, version='unlabeled'):
        self.name = 'SVD'
        self.ground_truth = self.load_groundtruth('datasets/test_groundtruth')
        self.unlabeled_keys = self.get_unlabeled_keys('datasets/unlabeled-data-id')
        if version == 'labeled':
            self.unlabeled_keys = []
        self.database = []
        for k, v in self.ground_truth.items():
            self.database.extend(list(map(str, v.keys())))
        self.queries = sorted(list(map(str, self.ground_truth.keys())))
        self.database += self.unlabeled_keys
        self.database = sorted(self.database)

    def load_groundtruth(self, filepath):
        gnds = OrderedDict()
        with open(filepath, 'r') as fp:
            for idx, lines in enumerate(fp):
                tmps = lines.strip().split(' ')
                qid = tmps[0]
                cid = tmps[1]
                gt = int(tmps[-1])
                if qid not in gnds:
                    gnds[qid] = {cid: gt}
                else:
                    gnds[qid][cid] = gt
        return gnds

    def get_unlabeled_keys(self, filepath):
        videos = list()
        with open(filepath, 'r') as fp:
            for tmps in fp:
                videos.append(tmps.strip())
        return videos

    def get_queries(self):
        return self.queries

    def get_database(self):
        return self.database

    def evaluate(self, similarities, all_db=None, verbose=True):
        mAP = []
        not_found = len(self.ground_truth.keys() - similarities.keys())
        for query, targets in self.ground_truth.items():
            y_true, y_score = [], []
            res = similarities[query]
            if isinstance(res, (np.ndarray, np.generic)):
                res = {v: s for v, s in zip(self.database, res) if v in all_db}
            for target, label in targets.items():
                if target in all_db:
                    s = res[target]
                    y_true.append(label)
                    y_score.append(s)

            for target in self.unlabeled_keys:
                if target in all_db:
                    s = res[target]
                    y_true.append(0)
                    y_score.append(s)
            mAP.append(average_precision_score(y_true, y_score))
        if verbose:
            print('=' * 5, 'SVD Dataset', '=' * 5)
            if not_found > 0:
                print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
            print('Database: {} videos'.format(len(all_db)))

            print('-' * 16)
            print('mAP: {:.4f}'.format(np.mean(mAP)))
        return {'mAP': np.mean(mAP)}


class VCSL(object):
    def __init__(self, datafolder, split='val', video_root=None):

        # label_json_file = 'label_file_uuid_total.json'
        self.name = 'VCSL'
        self.annotation = {}
        self.split = split
        self.video_root = video_root
        # with open(os.path.join(datafolder, label_json_file), 'r') as f:
        #     lines = f.readlines()
        #     gt_dict = json.loads(''.join(lines))
        #     self.annotation = gt_dict
        self.pair_file = os.path.join(datafolder, 'pair_file_' + split + '.csv')
        self.df = pd.read_csv(self.pair_file)
        # data_list = df[['query_id', 'reference_id']].values.tolist()
        self.split_meta_file = os.path.join(datafolder, 'split_meta_pairs.json')
        self.anno_file = os.path.join(datafolder, 'label_file_uuid_total.json')
        self.frames_all_file = os.path.join(datafolder, 'frames_all.csv')
        self.clip_gt = json.load(open(self.anno_file))
        if video_root is not None:
            self.all_data_file_list = [os.path.join(self.video_root, video_file) for video_file in
                                       os.listdir(self.video_root)]
            self.all_data_id_list = [video_path.split('/')[-1].split('.')[0].split('-')[0] for video_path in
                                     self.all_data_file_list]

        query_set = set()
        database_set = set()
        if split == 'all':
            pass  # todo
        else:
            with open(self.split_meta_file, 'r') as f:
                lines = f.readlines()
                split_meta_dict = json.loads(''.join(lines))
                for query, pair_list in split_meta_dict[split].items():
                    query_set.add(query)
                    query_gt_set = set()
                    for pair in pair_list:
                        database_set.add(pair.split('-')[0])
                        database_set.add(pair.split('-')[1])
                        query_gt_set.add(pair.split('-')[0])
                        query_gt_set.add(pair.split('-')[1])
                    self.annotation[query] = query_gt_set
        self.queries = sorted(list(query_set))
        self.database = sorted(list(database_set))

    def get_queries(self):
        return self.queries

    def get_database(self):
        return self.database

    def get_pairs(self, pair_file=None):
        if pair_file is not None:
            self.pair_file = pair_file
            self.df = pd.read_csv(self.pair_file)
        pairs = self.df[['query_id', 'reference_id']].values.tolist()
        return pairs

    def get_files_dict(self):
        files_dict = pd.read_csv(self.frames_all_file, usecols=['uuid', 'path'], index_col='uuid')
        files_dict = {idx: r['path'] for idx, r in files_dict.iterrows()}
        return files_dict

    def calculate_mAP(self, query, res, all_db):
        gt_sets = self.annotation[query]
        query_gt = gt_sets.intersection(all_db)

        i, ri, s = 0.0, 0, 0.0
        for video in sorted(res.keys(), key=lambda x: res[x], reverse=True):
            if video != query and video in all_db:
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri
        return s / len(query_gt)

    def evaluate(self, pred_file, metric):
        if metric.lower() == 'f1':
            self.evaluate_f1(pred_file)
        elif metric.lower() == 'map':
            similarities = json.load(open(pred_file))
            self.evaluate_map(similarities)

    def evaluate_map(self, similarities, all_db=None, verbose=True):
        if all_db is None:
            all_db = set()
            for query, refs in similarities.items():
                all_db.add(query)
                for ref in refs:
                    all_db.add(ref)
        all_db = set(self.database).intersection(all_db)

        all_query_mAP_list = []

        tmp_dict = {}
        for query, res in similarities.items():

            if query in self.queries:
                if isinstance(res, (np.ndarray, np.generic)):
                    res = {v: s for v, s in zip(self.database, res) if v in all_db}
                tmp_dict[query] = res
                all_query_mAP_list.append(self.calculate_mAP(query, res, all_db))
            # with open('cg_sim_from_dns_2.json', 'w') as f:
            #     f.write(json.dumps(tmp_dict))
        if verbose:
            print('=' * 5, 'VCSL Dataset' + '=' * 5)
            query_not_found = len(set(self.queries) - similarities.keys())
            if query_not_found > 0:
                print(f'[WARNING] Ideal count of queries is {len(set(self.queries))}, '
                      f'Actually using {len(similarities)}, and {query_not_found} queries are missing and will be ignored.')
            database_not_found = len(set(self.database)) - len(all_db)
            if database_not_found > 0:
                print(f'[WARNING] Ideal count of all data is {len(set(self.database))}, '
                      f'Actually using {len(all_db)}, and {database_not_found} datas are missing and will be ignored.')

            print('Queries: {} videos'.format(len(similarities)))
            print('Database: {} videos'.format(len(all_db)))

            print('-' * 16)
            print('mAP: {:.4f}'.format(np.mean(all_query_mAP_list)))
        return {'mAP': np.mean(all_query_mAP_list)}

    def evaluate_f1(self, pred_file):
        key_list = [key for key in self.clip_gt]

        meta_pairs = json.load(open(self.split_meta_file))

        # root_dir = os.path.dirname(self.anno_file)
        # split_file = os.path.join(root_dir, f"pair_file_{args.split}.csv")
        # df = pd.read_csv(split_file)
        split_pairs = set([f"{q}-{r}" for q, r in zip(self.df.query_id.values, self.df.reference_id.values)])
        print(f"{self.split} contains pairs {len(split_pairs)}")

        key_list = [key for key in key_list if key in split_pairs]
        print(f"Copied video data (positive) to evaluate: {len(key_list)}")

        config = dict()
        reader = build_reader('local', "json", **config)

        pred_dict = reader.read(pred_file)
        eval_list = []
        not_in_pred_key_num = 0
        for key in split_pairs:
            if key not in pred_dict.keys():
                not_in_pred_key_num += 1
                # print('not_in_pred_ key = ', key)
                continue
            if key in self.clip_gt:
                eval_list += [{"name": key, "gt": self.clip_gt[key], "pred": pred_dict[key]}]
            else:
                eval_list += [{"name": key, "gt": [], "pred": pred_dict[key]}]
        print(' not in pred key num = ', not_in_pred_key_num)
        print(f"finish loading files, start evaluation...")

        process_pool = Pool(4)
        result_list = process_pool.map(run_eval, eval_list)
        # print('result_list[0] =', result_list[0])
        result_dict = {i['name']: i for i in result_list}

        if self.split != 'all':
            meta_pairs = meta_pairs[self.split]
        else:
            meta_pairs = {**meta_pairs['train'], **meta_pairs['val'], **meta_pairs['test']}

        try:
            feat, vta = pred_file.split('-')[:2]
        except:
            feat, vta = 'My-FEAT', 'My-VTA'

        # Evaluated result on all video pairs including positive and negative copied pairs.
        # The segment-level precision/recall can also indicate video-level performance since
        # missing or false alarm lead to decrease on segment recall or precision.
        r, p, frr, far = evaluate_micro(result_dict, 1)  # todo
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

        r, p, cnt = evaluate_macro(result_dict, meta_pairs)

        print(f"query set cnt {cnt}, "
              f"query macro-Recall: {r:.2%}, "
              f"query macro-Precision: {p:.2%}, "
              f"F1: {2 * r * p / (r + p):.2%}, "
              )


class MPAA(object):
    def __init__(self, video_root, hdf5_file=None):
        self.name = 'MPAA'
        self.video_root = video_root
        self.master = os.path.join(self.video_root, 'master')
        self.all_root = os.path.join(self.video_root, 'all')
        self.all_data_file_list = [os.path.join(self.master, video_file) for video_file in os.listdir(self.master)] \
                                  + [os.path.join(self.all_root, video_file) for video_file in
                                     os.listdir(self.all_root)]
        self.all_data_id_list = [path.split(os.path.sep)[-1] for path in self.all_data_file_list]
        print('len of query_root = ', len(os.listdir(self.master)))
        print('len of all_root = ', len(os.listdir(self.all_root)))
        print('len of all_data_file_list = ', len(self.all_data_file_list))



        self.queries = [file for file in os.listdir(self.all_root) if not (file.endswith('txt') or file.endswith('idx') or file.endswith('sh') or file.endswith('out')) ]
        self.database = [file for file in os.listdir(self.master) if not (file.endswith('txt') or file.endswith('idx') or file.endswith('sh') or file.endswith('out'))]

        if hdf5_file is not None:
            feature_file = h5py.File(hdf5_file, "r")
            feature_keys = feature_file.keys()
            print(feature_keys)
            for fk in feature_keys:
                print(fk)
                print(feature_file[fk][:].shape)
            self.queries = [file for file in os.listdir(self.all_root) if file in feature_keys]
            self.database = [file for file in os.listdir(self.master) if file in feature_keys]

            print(f'\nLen of self.query in hdf5 file = {len(self.queries)}'
                  f', while all folder contains {len(os.listdir(self.all_root))}. '
                  f'\nThere are {len(os.listdir(self.all_root)) - len(self.queries)} folder file not in hdf5, and they are:')
            pprint([file for file in os.listdir(self.all_root) if file not in self.queries])

            print(f'len of self.database in hdf5 file = {len(self.database)},'
                  f'while master folders contains {len(os.listdir(self.master))}. '
                  f'\nThere are {len(os.listdir(self.master)) - len(self.database)} folder file not in hdf5, and they are:')
            pprint([file for file in os.listdir(self.master) if file not in self.database])
        self.trick_format_dict = {
            'slowmotion': 'slowmotion',
            'mashup': 'mashup',
            'tvepisodecam': 'tvepisodecam',
            'dark': 'dark',
            'tvclip': 'tvclip',
            'kbps': r'kbps|Kbps',
            'rotation': 'rotation',
            'different w*h': r'\d+x\d+',
            'withsubs': 'withsubs',
            'graphicoverlay': 'graphicoverlay',
            'segmentremoval': 'segmentremoval',
            'blur': r'blur|Blur',
            'Cropping': 'Cropping',
            'Scaling_HalfRes': 'Scaling_HalfRes',
            'MirroredHorizontally': 'MirroredHorizontally',
            'blackboxinsertion': 'blackboxinsertion',
            'ModifyContrast': 'ModifyContrast',
            'digest': 'digest',
            'MSUNoiseGenerator': 'MSUNoiseGenerator',
            'frameremoval': 'frameremoval',
            'rip': 'rip',
            'echo': 'echo',
            'findedges': 'findedges',
            'flip': 'flip',
            'insertclip': 'insertclip',
            'mosaic': 'mosaic',
            'noise': 'noise',
            'replicate': 'replicate',
            'rgb': 'rgb',
            'rotate': 'rotate',
            'scale': 'scale',
            'transition': 'transition',
            'gaussian': 'gaussian',
            'overlay': 'overlay',
            'contrast': 'contrast',
            'ghosting': 'ghosting',
            'lightning': 'lightning',
            'trailer': 'trailer',
        }
        self.special_trick_dict = {
            'other': self._match_other_files,
            'without_trailer': self._match_all_without_trailer
        }

    def _match_other_files(self, file_name):
        match = True
        for trick_name, trick_pattern in self.trick_format_dict.items():
            if match_pattern(file_name, trick_pattern):
                match = False
        return match

    def _match_all_without_trailer(self, file_name):
        match = True
        for trick_name, trick_pattern in self.trick_format_dict.items():
            if trick_name == 'trailer' and match_pattern(file_name, trick_pattern):
                match = False
        return match


    def get_queries(self):
        return self.queries

    def get_database(self):
        return self.database

    def get_pairs(self):
        return []

    def get_files_dict(self):
        return {}

    def evaluate(self, pred_file, metric='f1', filter_thresh=0):
        if metric.lower() == 'f1':
            return self.evaluate_f1(pred_file, filter_thresh=filter_thresh)


    def get_each_trick_dict(self, json_dict):
        each_trick_pred_dict = defaultdict(dict)
        for pair_str, pre_value in json_dict.items():
            query_file_name, ref_file_name = get_mpaa_video_file_path(pair_str, return_all_path=False)
            for trick_name, trick_pattern in self.trick_format_dict.items():
                if match_pattern(query_file_name, trick_pattern):
                    each_trick_pred_dict[trick_name][pair_str] = pre_value
            for trick_name, trick_func in self.special_trick_dict.items():
                if trick_func(query_file_name):
                    each_trick_pred_dict[trick_name][pair_str] = pre_value
        return each_trick_pred_dict

    def evaluate_f1(self, pred_file, filter_thresh=0):
        all_file_list = os.listdir(self.all_root)

        for trick_name, trick_pattern in self.trick_format_dict.items():
            file_num = 0
            for file_name in all_file_list:
                if match_pattern(file_name, trick_pattern):
                    file_num += 1
            print(f'In all folder, trick_name: "{trick_name}" file num = {file_num}')

        for trick_name, trick_func in self.special_trick_dict.items():
            file_num = 0
            for file_name in all_file_list:
                if trick_func(file_name):
                    file_num += 1
            print(f'In all folder, trick_name: "{trick_name}" file num = {file_num}')

        self.gt_dict = json.load(open('./mpaa_ground_truth/gt_json.json'))
        all_trick_gt_dict = self.get_each_trick_dict(self.gt_dict)
        print('pred_file = ', pred_file)
        pred_dict = json.load(open(pred_file))

        all_res_dict = dict()
        # get pred_dict for each copy trick
        all_trick_pred_dict = self.get_each_trick_dict(pred_dict)
        for trick_name, trick_pred_dict in all_trick_pred_dict.items():
            all_res_dict[trick_name] = self.evaluate_pred_dict_f1(trick_pred_dict, all_trick_gt_dict[trick_name],
                                                                  name=trick_name,
                                                                  save_trick_prefix=
                                                                  os.path.basename(pred_file).split('.')[0],
                                                                  filter_thresh=filter_thresh)

        print('=' * 100)
        all_res_dict['all'] = self.evaluate_pred_dict_f1(pred_dict, self.gt_dict, name='all',
                                                         filter_thresh=filter_thresh)
        print('\n' + '=' * 100)
        return all_res_dict

    def filter_gt(self, pred_dict, gt_dict):
        filtered_gt_dict = dict()
        pred_not_in_gt = 0
        for gt_pair_str, gt_value in gt_dict.items():
            if gt_pair_str in pred_dict:
                filtered_gt_dict[gt_pair_str] = gt_value
            else:
                pred_not_in_gt += 1
                # print(f'gt_pair_str "{gt_pair_str}" not in pred_dict')
        # print(f'number of pred not in gt = {pred_not_in_gt}')
        return filtered_gt_dict

    def evaluate_pred_dict_f1(self, pred_dict, gt_dict, name='all', save_trick_prefix=None, filter_thresh=0):
        print('\n' + '=' * 100)
        print(f'for {name} case, result is:')
        filtered_gt_dict = self.filter_gt(pred_dict, gt_dict)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for pred_pair_str, pred_value in pred_dict.items():
            if pred_value is None or len(pred_value) == 0 \
                    or filter_short_pred(pred_value, filter_thresh=filter_thresh):
                if len(filtered_gt_dict[pred_pair_str]) > 0:
                    FN += 1
                else:
                    TN += 1
            else:
                if len(filtered_gt_dict[pred_pair_str]) > 0:
                    TP += 1
                else:
                    FP += 1

        print(f'TP = {TP}, TN = {TN}, FP = {FP}, FN = {FN}')
        print(f'TP + TN + FP + FN = {TP + TN + FP + FN}')
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0
        frr = FN / (FN + TP) if (FN + TP) != 0 else 0
        far = FP / (FP + TN) if (FP + TN) != 0 else 0
        print(f"Overall segment-level performance, "
              f"Recall: {recall:.2%}, "
              f"Precision: {precision:.2%}, "
              f"F1: {f1:.2%}, "
              )
        print(f"video-level performance, "
              f"FRR: {frr:.2%}, "
              f"FAR: {far:.2%}, "
              )
        if save_trick_prefix is not None:
            with open(f'./mpaa_ground_truth/each_trick/{save_trick_prefix}-{name}-gt.json', 'w') as f:
                f.write(json.dumps(gt_dict, indent=4))
            with open(f'./mpaa_ground_truth/each_trick/{save_trick_prefix}-{name}-pred.json', 'w') as f:
                f.write(json.dumps(pred_dict, indent=4))
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'frr': frr,
            'far': far
        }


class MUSCLE_VCD(object):
    def __init__(self, video_root, query='st2'):
        self.name = 'MUSCLE_VCD'
        self.video_root = video_root
        self.master = os.path.join(self.video_root, 'master')
        self.st1 = os.path.join(self.video_root, 'st1')
        self.st2 = os.path.join(self.video_root, 'st2')
        self.all_data_file_list = [os.path.join(self.master, video_file) for video_file in os.listdir(self.master)] \
                                  + [os.path.join(self.st1, video_file) for video_file in os.listdir(self.st1)] \
                                  + [os.path.join(self.st2, video_file) for video_file in os.listdir(self.st2)]
        self.all_data_id_list = [path.split(os.path.sep)[-1] for path in self.all_data_file_list]
        print('len of master = ', len(os.listdir(self.master)))
        print('len of st1 = ', len(os.listdir(self.st1)))
        print('len of st2 = ', len(os.listdir(self.st2)))
        print('len of all_data_file_list = ', len(self.all_data_file_list))

        self.queries = [file for file in os.listdir(getattr(self, query)) if file.endswith('mpg')]
        self.database = [file for file in os.listdir(self.master) if file.endswith('mpg')]

    def get_queries(self):
        return self.queries

    def get_database(self):
        return self.database

    def get_pairs(self):
        return []

    def get_files_dict(self):
        return {}

    def get_p_n_ratio(self, gt_dict, pred_dict=None):
        pos_num = 0
        neg_num = 0
        for pair_name, pred_t_list in gt_dict.items():
            if pred_dict is not None and pair_name not in pred_dict.keys():
                continue
            if pred_t_list is None or len(pred_t_list) == 0:
                neg_num += 1
            else:
                pos_num += 1
        return pos_num / neg_num

    def evaluate(self, pred_file, metric='f1'):
        if metric.lower() == 'f1':
            self.evaluate_f1(pred_file)
        elif metric.lower() == 'map':
            similarities = json.load(open(pred_file))
            self.evaluate_map(similarities)

    def evaluate_f1(self, pred_file):
        self.clip_gt = json.load(open(os.path.join(self.st2, 'gt_json.json')))

        config = dict()
        reader = build_reader('local', "json", **config)

        pred_dict = reader.read(pred_file)
        eval_list = []
        not_in_pred_key_num = 0
        for key in pred_dict.keys():
            if key not in pred_dict.keys():
                not_in_pred_key_num += 1
                # print('not_in_pred_ key = ', key)
                continue
            if key in self.clip_gt:
                eval_list += [{"name": key, "gt": self.clip_gt[key], "pred": pred_dict[key]}]
            else:
                eval_list += [{"name": key, "gt": [], "pred": pred_dict[key]}]
        print(' not in pred key num = ', not_in_pred_key_num)
        print(f"finish loading files, start evaluation...")

        process_pool = Pool(4)
        result_list = process_pool.map(run_eval, eval_list)
        result_dict = {i['name']: i for i in result_list}

        meta_pairs = {}
        for key in self.clip_gt:
            q_id = key.split('-')[0]
            if q_id not in meta_pairs.keys():
                meta_pairs[q_id] = [key]
            else:
                meta_pairs[q_id].append(key)
        try:
            feat, vta = pred_file.split('-')[:2]
        except:
            feat, vta = 'My-FEAT', 'My-VTA'

        # Evaluated result on all video pairs including positive and negative copied pairs.
        # The segment-level precision/recall can also indicate video-level performance since
        # missing or false alarm lead to decrease on segment recall or precision.
        r, p, frr, far = evaluate_micro(result_dict, self.get_p_n_ratio(self.clip_gt))  # todo
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

        r, p, cnt = evaluate_macro(result_dict, meta_pairs)

        print(f"query set cnt {cnt}, "
              f"query macro-Recall: {r:.2%}, "
              f"query macro-Precision: {p:.2%}, "
              f"F1: {2 * r * p / (r + p):.2%}, "
              )
        # slowmotion mashup tvepisodecam dark tvclip kbps rotation different withsubs graphicoverlay segmentremoval blur Cropping Scaling_HalfRes MirroredHorizontally blackboxinsertion ModifyContrast digest MSUNoiseGenerator frameremoval rip echo findedges flip insertclip mosaic noise replicate rgb rotate scale transition gaussian overlay contrast ghosting lightnin
