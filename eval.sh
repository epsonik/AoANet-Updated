CUDA_VISIBLE_DEVICES=0

python -u eval.py --model log/old/resnet152/log_aoanet/model-best.pth \
    --infos_path log/old/resnet152/log_aoanet/infos_aoanet.pkl \
    --dump_images 0 \
    --dump_json 1 \
    --num_images -1 \
    --language_eval 1 \
    --beam_size 3 \
    --batch_size 400 \
    --split test \
    --cnn_model regnet16 \
