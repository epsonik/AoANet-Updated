python eval.py --model /home/bartosiewicz/mateusz/densenet121/log_aoanet/model-best.pth \
--infos_path /home/bartosiewicz/mateusz/densenet121/log_aoanet/infos_aoanet-best.pkl \
--input_json /home/bartosiewicz/mateusz/AoANet-Updated/data/cocotalk.json \
--input_fc_dir /home/bartosiewicz/mateusz/AoANet-Updated/data/densenet121_fc \
--input_att_dir /home/bartosiewicz/mateusz/AoANet-Updated/data/densenet121_att \
--input_label_h5 none \
--split val \
--image_id 285645 \
--batch_size 1