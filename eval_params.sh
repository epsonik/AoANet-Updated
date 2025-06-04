#!/bin/bash
CUDA_VISIBLE_DEVICES=0

python -u eval_params.py --model log/old/regnet/log_aoanet/model-best.pth \
    --infos_path log/old/regnet/log_aoanet/infos_aoanet.pkl \
    --dump_images 0 \
    --dump_json 1 \
    --num_images -1 \
    --name regnet \
    --language_eval 1 \
    --beam_size 2 \
    --batch_size 40 \
    --split test \
    --cnn_model regnet16 ;
