#!/bin/bash
CUDA_VISIBLE_DEVICES=0
for b in regnet;
do
python -u eval_params.py --model log/old/$b/log_aoanet/model-best.pth \
    --infos_path log/old/$b/log_aoanet/infos_aoanet.pkl \
    --dump_images 0 \
    --dump_json 1 \
    --num_images -1 \
    --name $b \
    --language_eval 1 \
    --beam_size 2 \
    --batch_size 40 \
    --split test \
    --cnn_model regnet16 ;
done
