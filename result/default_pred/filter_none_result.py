import json
import os

def _change_s_format(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    res = "%02d:%02d:%02d" % (h, m, s)
    return res


def filter_json_result(json_file):
    pre_dict = json.load(open(json_file))
    new_dict = dict()
    for k, v in pre_dict.items():
        if len(v) != 0:
            new_dict[k] = v
    print(f'before filter, len(pre_dict)={len(pre_dict)}')
    print(f'after filter, len(new_dict)={len(new_dict)}')
    old_file_name = os.path.split(json_file)[-1].split('.')[0]
    new_file_name = old_file_name + 'filtered.json'
    file_path = os.path.dirname(json_file)
    new_path = os.path.join(file_path, new_file_name)
    print('new path = ', new_path)
    with open(new_path, 'w') as f:
        f.write(json.dumps(new_dict, indent=4))

def change_time_format(json_file):
    pre_dict = json.load(open(json_file))
    new_dict = dict()
    for k, v in pre_dict.items():
        new_v = []
        for clip_time_infos in v:
            new_clip_time_infos = []
            for t in clip_time_infos:
                new_t = _change_s_format(t)
                new_clip_time_infos.append(new_t)
            new_v.append(new_clip_time_infos)
        new_dict[k] = new_v

    old_file_name = os.path.split(json_file)[-1].split('.')[0]
    new_file_name = old_file_name + '_time_formatted.json'
    file_path = os.path.dirname(json_file)
    new_path = os.path.join(file_path, new_file_name)
    print('new path = ', new_path)
    with open(new_path, 'w') as f:
        f.write(json.dumps(new_dict, indent=4))

if __name__ == '__main__':
    # filter_json_result(
    #     './result/default_pred/mpaa-dns_backbone-qd_pair-dns_sim-TN-pred.json')

    change_time_format('./result/default_pred/mpaa-dns_backbone-qd_pair-dns_sim-TN-pred_filtered.json')
