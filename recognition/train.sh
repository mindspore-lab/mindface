# extract images
python src/rec2jpg_dataset.py --include /home/data/xieguochen/dataset/AgeDataset/faces_webface_112x112 \
 --output /home/data/xieguochen/dataset/AgeDataset/faces_webface_112x112_train

# MS1M V2
sh scripts/run_distribute_train_gpu.sh  configs/train_config_ms1m.yaml 2

# CASIA
sh scripts/run_distribute_train_gpu.sh  configs/train_config_casia.yaml 2

# Eval
sh scripts/run_eval_gpu.sh /home/data/xieguochen/dataset/AgeDataset/faces_emore train_parallel_small/ArcFace--12_802.ckpt R50

