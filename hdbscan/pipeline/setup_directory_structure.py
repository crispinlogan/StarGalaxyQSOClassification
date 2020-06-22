import os
from config import ConfigVars

for test_bool in [True, False]:
    conf = ConfigVars(test = test_bool)
    os.makedirs(f'{conf.base_path}/{conf.data_output}', exist_ok=True)
    os.makedirs(conf.RF_dir, exist_ok=True)
    os.makedirs(conf.hdb_dir, exist_ok=True)
    os.makedirs(conf.sav_mod_dir, exist_ok=True)
    os.makedirs(conf.cons_dir, exist_ok=True)
    os.makedirs(conf.pred_dir, exist_ok=True)
