#!/bin/bash
CUDA_VISIBLE_DEVICES=3
python eval.py --model /home/bartosiewicz/mateusz/densenet121/log_aoanet/model-best.pth \
    --infos_path /home/bartosiewicz/mateusz/densenet121/log_aoanet/infos_aoanet-best.pkl \
    --dump_json 1 \
    --name single_eval_raw \
    --beam_size 3 \
    --batch_size 40 \
    --cnn_model densenet121 \
    --image_folder  vis \
    --coco_json eval_image.json