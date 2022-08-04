# video_decopy_detection

### requirements
```
tslearn
```

### Feature extracting
```shell
python feature_extracting.py --dataset VCSL --feature_backbone DnS_R50 --output_type hdf5 --output_name ./features/my_features.hdf5 --video_root /home/zhangchen/zhangchen_workspace/dataset/VCSL
```
    
### Calculate similarity matrix
query-database-wise without pair_file, DnS similarity
```shell
python calcu_similarity_matrix.py --dataset VCSL --feature_path ./features/vcsl_feature.hdf5 --similarity_type DnS --dns_student_type attention --output_dir ./sim_matrix_npy-without_pairs-dns_sim
```
using pair_file, DnS similarity
```shell
python calcu_similarity_matrix.py --dataset VCSL --feature_path ./features/vcsl_feature.hdf5 --similarity_type DnS --pair_file ./vcsl_data/pair_file_val.csv --dns_student_type attention --output_dir ./sim_matrix_npy-val_pairs-dns_sim
```

query-database-wise without pair_file, cos similarity
```shell
python calcu_similarity_matrix.py --dataset VCSL --feature_path ./features/vcsl_feature.hdf5 --similarity_type cos --output_dir ./sim_matrix_npy-without_pairs-cos_sim
```
using pair_file, cos similarity
```shell
python calcu_similarity_matrix.py --dataset VCSL --feature_path ./features/vcsl_feature.hdf5 --similarity_type cos --pair_file ./vcsl_data/pair_file_val.csv --output_dir ./sim_matrix_npy-val_pairs-cos_sim
```

### Evaluate metrics
F1 metric:
```shell
python evaluate.py --dataset VCSL --pred_file ./result/output/dino-DTW-pred.json --split test --metric f1
```
mAP metric:
```shell
python evaluate.py --dataset VCSL --pred_file ./result/output/sim_vcsl.json --split val --metric map
```


### Video temporal alignment
```shell
python temporal_alignment.py \
        --pair-file ./vcsl_data/pair_file_val.csv \
        --input-root /home/zhangchen/zhangchen_workspace/video_decopy_detection/sim_matrix_npy \
        --input-store local \
        --batch-size 32 \
        --data-workers 32 \
        --request-workers 16 \
        --alignment-method DTW \
        --output-root result/best/ \
        --result-file dino-DTW-pred.json \
        --params-file result/vcsl/tune/dino-DTW-21112318/result.json
```