#!/bin/bash
CUDA_VISIBLE_DEVICES=0
for k in 3 2 5 8 1;
do
python -u eval.py --model log/log_aoanet/model-best.pth \
    --infos_path log/log_aoanet/infos_aoanet-best.pkl \
    --dump_images 0 \
    --dump_json 1 \
    --num_images -1 \
    --name densenet121_f$k \
    --language_eval 1 \
    --beam_size $k \
    --batch_size 40 \
    --split test \
    --cnn_model regnet16 ;
done