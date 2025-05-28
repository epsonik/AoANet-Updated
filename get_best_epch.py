from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import opts
import argparse
import misc.utils as utils


infos_paths=["log/old/regnet16/log_aoanet/infos_aoanet-best.pkl"]
infos_paths_rl=["log/old/regnet16/log_aoanet_rl/infos_aoanet-best.pkl"]

for x in infos_paths:

    infos={}
    with open(x, 'rb') as f:
        infos = utils.pickle_load(f)
    print(x)
    print(infos.get('epoch', 0))

for x in infos_paths_rl:
    infos = {}
    with open(x, 'rb') as f:
        infos = utils.pickle_load(f)
    print(x)
    print(infos.get('epoch', 0))



