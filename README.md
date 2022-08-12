# video_decopy_detection



### Step1.Feature extracting
```shell
python feature_extracting.py \
        --dataset VCSL \
        --feature_backbone DnS_R50 \
        --output_type hdf5 \
        --output_name ./features/my_features.hdf5 \
        --video_root /home/zhangchen/zhangchen_workspace/dataset/VCSL
```
    
### Step2.Calculate similarity matrix
query-database-wise without pair_file, DnS similarity
```shell
python calcu_similarity_matrix.py \
        --dataset VCSL \
        --feature_path ./features/vcsl_feature.hdf5 \
        --similarity_type DnS \
        --dns_student_type attention \
        --output_dir ./sim_matrix_npy/vcsl-dns_backbone-qd_pair-dns_sim
```
using pair_file, DnS similarity
```shell
python calcu_similarity_matrix.py \
        --dataset VCSL \
        --feature_path ./features/vcsl_feature.hdf5 \
        --similarity_type DnS \
        --pair_file ./vcsl_data/pair_file_val.csv \
        --dns_student_type attention \
        --output_dir ./sim_matrix_npy/vcsl-dns_backbone-val_pairs-dns_sim
```

query-database-wise without pair_file, cos similarity
```shell
python calcu_similarity_matrix.py \
        --dataset VCSL \
        --feature_path ./features/vcsl_feature.hdf5 \
        --similarity_type cos \
        --output_dir ./sim_matrix_npy/vcsl-dns_backbone-qd_pair-cos_sim
```
using pair_file, cos similarity
```shell
python calcu_similarity_matrix.py \
        --dataset VCSL \
        --feature_path ./features/vcsl_feature.hdf5 \
        --similarity_type cos \
        --pair_file ./vcsl_data/pair_file_val.csv \
        --output_dir ./sim_matrix_npy/vcsl-dns_backbone-val_pairs-cos_sim
```

### Step3.Video temporal alignment
tune params.
```shell
python temporal_alignment_tune.py \
        --pair_file ./vcsl_data/pair_file_val.csv \
        --input_root ./sim_matrix_npy/vcsl-dns_backbone-val_pairs-dns_sim \
        --batch_size 32 \
        --data_workers 32 \
        --request_workers 16 \
        --alignment_method DTW \
        --output_workers 16 \
        --output_root ./result/tune/vcsl-dns_backbone-val_pairs-dns_sim-DTW/ \
        --tn_max_step="5:15:5" \
        --tn_top_K="5:15:5" \
        --min_sim="0.2:0.31:0.1" \
        --discontinue="9:11:1" \
        --sum_sim="-2:10:1" \
        --diagonal_thres="10:50:10" \
        --ave_sim="1.1:1.31:0.1"

```

use tuned param file `./result/tune/vcsl-dns_backbone-val_pairs-dns_sim-DTW/result.json`, to output the pred file `./result/best_pred/vcsl-dns_backbone-val_pairs-dns_sim-DTW-pred.json`.
```shell
python temporal_alignment.py \
        --pair_file ./vcsl_data/pair_file_val.csv \
        --input_root ./sim_matrix_npy/vcsl-dns_backbone-val_pairs-dns_sim \
        --batch_size 32 \
        --data_workers 32 \
        --request_workers 16 \
        --alignment_method DTW \
        --output_root ./result/best_pred/ \
        --result_file vcsl-dns_backbone-val_pairs-dns_sim-DTW-pred.json \
        --params_file ./result/tune/vcsl-dns_backbone-val_pairs-dns_sim-DTW/result.json
```

to use default param, just ignore `--params_file`.  
to use default all query-database pairs, just ignore `--pair_file`.  
to use spd model, add `--spd-model-path data/spd_models/${FEAT}.pt`  and `--device cuda:0`

### Step4.Evaluate metrics
F1 metric:
```shell
python evaluate.py \
        --dataset VCSL \
        --pred_file ./result/best_pred/vcsl-dns_backbone-val_pairs-dns_sim-DTW-pred.json \
        --split val \
        --metric f1
```

[//]: # (mAP metric:)

[//]: # (```shell)

[//]: # (python evaluate.py --dataset VCSL --pred_file ./result/output/sim_vcsl.json --split val --metric map)

[//]: # (```)

### Step5.Visualization similarity matrix
```shell
python visualization.py \
        --sim_np_folder /home/zhangchen/zhangchen_workspace/video_decopy_detection/sim_matrix_npy/muscle-dns_backbone-st2_pair-cos_sim \
        --pred_file ./result/default_pred/muscle-dns_backbone-st2_pairs-cos_sim-TN-pred.json \
        --gt_file ./muscle_vcd/st2/gt_json.json \
        --save_dir ./visual_imgs/muscle-dns_backbone-st2_pairs-cos_sim-TN_default \
        --ignore_none_res true
```





