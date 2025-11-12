python visualize.py \
    --model /home/bartosiewicz/mateusz/regnet/log_aoanet/model-best.pth \
    --infos_path /home/bartosiewicz/mateusz/regnet/log_aoanet/infos_aoanet-best.pkl \
    --cnn_model regnet \
    --image_folder vis/imgs \
    --num_images -1 \
    --batch_size 3 \
    --output_dir vis/regnet_aoanet