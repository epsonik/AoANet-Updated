python scripts/prepro_feats.py --input_json '/home/bartosiewicz/mateusz/dane/coco2014/karpathy/cocotalk.json' --output_dir data/cocotalk --images_root '/home/bartosiewicz/mateusz/dane/coco2014'

python scripts/prepro_labels.py --input_json '/home/bartosiewicz/mateusz/dane/coco2014/karpathy/dataset_coco.json' --output_json data/cocotalk.json --output_h5 data/cocotalk