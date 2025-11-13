python visualize.py \
    --model /home/bartosiewicz/mateusz/resnet101/log_aoanet/model-best.pth \
    --infos_path /home/bartosiewicz/mateusz/resnet101/log_aoanet/infos_aoanet-best.pkl \
    --cnn_model resnet101 \
    --image_folder vis/imgs \
    --num_images 0\
    -- dump_images 1 \
    --batch_size 3\
    -- dump_path 1 \
    --output_dir vis/resnet101_aoanet \

python visualize.py \
    --model /home/bartosiewicz/mateusz/resnet152/log_aoanet/model-best.pth \
    --infos_path /home/bartosiewicz/mateusz/resnet152/log_aoanet/infos_aoanet-best.pkl \
    --cnn_model resnet152 \
    --image_folder vis/imgs \
    --num_images 0\
    --batch_size 3\
    -- dump_images 1 \
    -- dump_path 1 \
    --output_dir vis/resnet152_aoanet \
