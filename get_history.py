import misc.utils as utils
import os

g_path = "/mnt/dysk2/dane/AoANet-Updated/log/old/"

# config_list = ["densenet121","densenet161","densenet201","bottom_up", "inception","regnet","resnet101", "resnet152"]
config_list = ["densenet121"]
histories = {}
histories_rl = {}
for config_item in config_list:
    with open(os.path.join(g_path, config_item, 'log_aoanet', 'histories_aoanet.pkl'), 'rb') as f:
        histories = utils.pickle_load(f)
    with open(os.path.join(g_path, config_item, 'log_aoanet_rl', 'histories_aoanet.pkl'), 'rb') as f:
        histories_rl = utils.pickle_load(f)

val_result_history = histories.get('val_result_history', {})
loss_history = histories.get('loss_history', {})
lr_history = histories.get('lr_history', {})
ss_prob_history = histories.get('ss_prob_history', {})
print(val_result_history)
print(loss_history)
print(lr_history)
print(ss_prob_history)

val_result_history = histories_rl.get('val_result_history', {})
loss_history = histories_rl.get('loss_history', {})
lr_history = histories_rl.get('lr_history', {})
ss_prob_history = histories_rl.get('ss_prob_history', {})
print(val_result_history)
print(loss_history)
print(lr_history)
print(ss_prob_history)