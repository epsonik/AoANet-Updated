#!/bin/bash
CUDA_VISIBLE_DEVICES=0
for b in regnet densenet121 densenet161 densenet201 inception resnet152 resnet101 bottom_up;
do
python -u eval_params.py --model log/old/$b/log_aoanet_rl/model-best.pth \
    --infos_path log/old/$b/log_aoanet_rl/infos_aoanet.pkl \
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