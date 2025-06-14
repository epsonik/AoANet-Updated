import misc.utils as utils
import os
import csv
import os

g_path = "/mnt/dysk2/dane/AoANet-Updated/log/old/"

config_list = ["regnet"]

histories_rl = {}
for config_item in config_list:
    with open(os.path.join(g_path, config_item, 'log_aoanet_rl', 'histories_aoanet.pkl'), 'rb') as f:
        histories_rl = utils.pickle_load(f)

    val_result_history = histories_rl.get('val_result_history', {})
    loss_history = histories_rl.get('loss_history', {})

    lines_dict = []
    header = ["iteration", "CIDEr", "BLEU-4", "loss"]
    filename = os.path.join("/mnt/dysk2/dane/AoANet-Updated/log/", config_item + "_metrics.csv")
    for iteration in val_result_history.keys():
        val_CIDEr = val_result_history[iteration]['lang_stats']['CIDEr']
        val_BLEU_4 = val_result_history[iteration]['lang_stats']['Bleu_4']
        val_loss = val_result_history[iteration]['loss']
        lines_dict.append(
            {"iteration": iteration, "CIDEr": val_CIDEr, "BLEU-4": val_BLEU_4, "loss": val_loss})
    with open(filename, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(lines_dict)

    # lines_dict = []
    # header = ["iteration", "loss"]
    # filename = os.path.join("/mnt/dysk2/dane/AoANet-Updated/log/", config_item + "_loss.csv")
    # for iteration in loss_history.keys():
    #     loss = loss_history[iteration]
    #     if type(loss) != float:
    #         loss = loss.item()
    #     lines_dict.append(
    #         {"iteration": iteration, "loss": loss})
    #
    # with open(filename, 'a') as f:
    #     writer = csv.DictWriter(f, fieldnames=header)
    #     writer.writeheader()
    #     writer.writerows(lines_dict)
