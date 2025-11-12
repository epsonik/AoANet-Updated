python visualize.py \
    --model /home/bartosiewicz/mateusz/densenet201/log_aoanet/model-best.pth \
    --infos_path /home/bartosiewicz/mateusz/densenet201/log_aoanet/infos_aoanet-best.pkl \
    --cnn_model densenet201 \
    --image_folder vis/imgs \
    --num_images -1 \
    --batch_size 3 \
    --output_dir vis/densenet201_aoanet \

python visualize.py \
    --model /home/bartosiewicz/mateusz/densenet121/log_aoanet/model-best.pth \
    --infos_path /home/bartosiewicz/mateusz/densenet121/log_aoanet/infos_aoanet-best.pkl \
    --cnn_model densenet121 \
    --image_folder vis/imgs \
    --num_images -1 \
    --batch_size 3 \
    --output_dir vis/densenet121_aoanet \

python visualize.py \
    --model /home/bartosiewicz/mateusz/regnet/log_aoanet/model-best.pth \
    --infos_path /home/bartosiewicz/mateusz/regnet/log_aoanet/infos_aoanet-best.pkl \
    --cnn_model regnet \
    --image_folder vis/imgs \
    --num_images -1\
    --batch_size 3\
    --output_dir vis/regnet_aoanet \

python visualize.py \
    --model /home/bartosiewicz/mateusz/densenet161/log_aoanet/model-best.pth \
    --infos_path /home/bartosiewicz/mateusz/densenet161/log_aoanet/infos_aoanet-best.pkl \
    --cnn_model densenet161 \
    --image_folder vis/imgs \
    --num_images -1\
    --batch_size 3\
    --output_dir vis/densenet161_aoanet \

python visualize.py \
    --model /home/bartosiewicz/mateusz/inception/log_aoanet/model-best.pth \
    --infos_path /home/bartosiewicz/mateusz/inception/log_aoanet/infos_aoanet-best.pkl \
    --cnn_model inception \
    --image_folder vis/imgs \
    --num_images -1\
    --batch_size 3\
    --output_dir vis/inception_aoanet \


python visualize.py \
    --model /home/bartosiewicz/mateusz/resnet101/log_aoanet/model-best.pth \
    --infos_path /home/bartosiewicz/mateusz/resnet101/log_aoanet/infos_aoanet-best.pkl \
    --cnn_model resnet101 \
    --image_folder vis/imgs \
    --num_images -1\
    --batch_size 3\
    --output_dir vis/resnet101_aoanet \

python visualize.py \
    --model /home/bartosiewicz/mateusz/resnet152/log_aoanet/model-best.pth \
    --infos_path /home/bartosiewicz/mateusz/resnet152/log_aoanet/infos_aoanet-best.pkl \
    --cnn_model resnet152 \
    --image_folder vis/imgs \
    --num_images -1\
    --batch_size 3\
    --output_dir vis/resnet152_aoanet



#python visualize.py \
#    --model /home/bartosiewicz/mateusz/resnet101/log_aoanet/model-best.pth \
#    --infos_path /home/bartosiewicz/mateusz/resnet101/log_aoanet/infos_aoanet-best.pkl \
#    --cnn_model resnet101 \
#    --image_folder vis/imgs \
#    --num_images -1\
#    --batch_size 3\
#    --output_dir vis/resnet101_aoanet \
#
