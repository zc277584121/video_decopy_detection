# video_decopy_detection

### Feature extracting
```shell
python feature_extracting.py --dataset VCSL --feature_backbone DnS_R50 --output_type hdf5 --output_name ./features/my_features.hdf5 --video_root /home/zhangchen/zhangchen_workspace/dataset/VCSL
```
    
### Calculate similarity matrix
```shell
python calcu_similarity_matrix.py --dataset VCSL --feature_path ./features/vcsl_feature.hdf5 --similarity_type DnS --dns_student_type attention
```