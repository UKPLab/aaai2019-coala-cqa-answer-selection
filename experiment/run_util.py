import logging
import sys

import tensorflow as tf

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True


def setup_logger(config_logger, name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel(config_logger['level'])
    handler_stdout.setFormatter(formatter)
    logger.addHandler(handler_stdout)

    if 'path' in config_logger:
        handler_file = logging.FileHandler(config_logger['path'])
        handler_file.setLevel(config_logger['level'])
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    logger.setLevel(config_logger['level'])
    return logger
