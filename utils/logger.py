import logging
import os
import sys
import os.path as osp
import datetime

def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    train_log_name = "train_log_"+datetime.datetime.now().strftime('%m%d%H%M') + ".txt"
    test_log_name = "test_log_"+datetime.datetime.now().strftime('%m%d%H%M') + ".txt"
    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, train_log_name), mode='w')
        else:
            # fh = logging.FileHandler(os.path.join(save_dir, "test_log_"+datetime.datetime.now().strftime('%Y_%m_%d_%T') + ".txt"), mode='w')
            fh = logging.FileHandler(os.path.join(save_dir, test_log_name), mode='w')

        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger