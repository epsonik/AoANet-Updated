#!/bin/bash
CUDA_VISIBLE_DEVICES=0
for k in 2 3 5 8 1;
do
python -u eval.py --model log/old/densenet121/log_aoanet_rl/model-best.pth \
    --infos_path log/old/densenet121/log_aoanet_rl/infos_aoanet-best \
    --dump_images 0 \
    --dump_json 1 \
    --num_images -1 \
    --name bottom_up$k \
    --language_eval 1 \
    --beam_size $k \
    --batch_size 40 \
    --split test \
    --cnn_model regnet16 ;
done