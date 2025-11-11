#!/bin/bash
CUDA_VISIBLE_DEVICES=3
python -u eval.py --model /home/bartosiewicz/mateusz/log_aoanet/model-best.pth \
    --infos_path /home/bartosiewicz/mateusz/log_aoanet/infos_aoanet-best.pkl \
    --dump_images 0 \
    --dump_json 1 \
    --num_images -1 \
    --name resnet101_f3 \
    --language_eval 1 \
    --beam_size 3 \
    --batch_size 40 \
    --split test \
    --cnn_model regnet16