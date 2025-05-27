from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import opts
import argparse
import misc.utils as utils


infos_path="log/old/inception/log_aoanet/infos_aoanet-best.pkl"
infos={}
with open(infos_path, 'rb') as f:
    infos = utils.pickle_load(f)
print(infos_path)
print(infos.get('epoch', 0))
# for k in vars(infos['opt']).keys():
#     print(k)
#     print(getattr(infos['opt'], k, ''))


