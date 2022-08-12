import json
def change_2_second(time_str):
    time_str_list = time_str.split(':')
    return int(int(time_str_list[0]) * 3600 + int(time_str_list[1]) * 60 + int(time_str_list[2]))

if __name__ == '__main__':
    gt_dict = {}
    with open('/Users/zilliz/zilliz/video_decopy_detection/muscle_vcd/st2/ST2QueriesGrt.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_split_list = line.split(' ')
            query_id = line_split_list[0] + '.mpg'

            query_start_time = line_split_list[1]
            query_start_time_s = change_2_second(query_start_time)

            query_end_time = line_split_list[2]
            query_end_time_s = change_2_second(query_end_time)

            ref_id = line_split_list[3].lower() + '.mpg'

            ref_start_time = line_split_list[4]
            ref_start_time_s = change_2_second(ref_start_time)

            ref_end_time_s = ref_start_time_s + (query_end_time_s - query_start_time_s)

            # print(query_id)
            # print(query_start_time_s)
            # print(query_end_time_s)
            # print(ref_id)
            # print(ref_start_time_s)
            # print(ref_end_time_s)
            # print('')
            key = query_id + '-' + ref_id
            value = [query_start_time_s, ref_start_time_s, query_end_time_s, ref_end_time_s]
            gt_dict[key] = value
    print(gt_dict)
    
    pred_dict = json.load(open('/Users/zilliz/zilliz/video_decopy_detection/result/best_pred/muscle-dns_backbone-st2_pairs-dns_sim-DTW-pred.json'))
    for pred_k in pred_dict.keys():
        if pred_k in gt_dict.keys():
            pred_dict[pred_k] = [gt_dict[pred_k]]
        else:
            pred_dict[pred_k] = []
    trans_format_str = json.dumps(pred_dict)
    with open('/Users/zilliz/zilliz/video_decopy_detection/muscle_vcd/st2/gt_json.json', 'w') as f:
        f.write(trans_format_str)