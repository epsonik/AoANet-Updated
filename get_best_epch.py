from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import opts
import argparse
import misc.utils as utils


infos_path="log/old/inception/log_aoanet/infos_aoanet-best.pkl"
with open(infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

print(infos['opt'])
for k in vars(infos['opt']).keys():
   print(infos['opt'][k])
