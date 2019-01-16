import datetime
import importlib
import json
import logging
import os
from math import log, ceil, floor

import click
import numpy as np
import tensorflow as tf
import yaml

from experiment.config import load_config
from experiment.run_util import setup_logger, sess_config
from experiment.util import replace_dict_values


def exponential(min_val, max_val, base):
    return base ** np.random.uniform(log(min_val, base), log(max_val, base))


def normal(e, s):
    outcome = -1
    while outcome < 0:
        outcome = np.random.normal(e, s)
    return outcome


def half_normal(e, s):
    outcome = np.random.normal(e, s)
    if outcome < e:
        outcome = e + (e - outcome)
    return outcome


def uniform(min, max, stepsize=None):
    if stepsize is None:
        return np.random.uniform(min, max)
    else:
        possibilities = floor((max - min) / stepsize)
        return min + np.random.randint(0, possibilities + 1) * stepsize


sampling = {
    'uniform': lambda conf: uniform(conf['min'], conf['max'], conf.get('stepsize')),
    'uniform_choice': lambda conf: conf['options'][np.random.randint(0, len(conf['options']))],
    'exponential': lambda conf: exponential(conf['min'], conf['max'], conf['base']),
    'normal': lambda conf: normal(conf['e'], conf['s']),
    'half_normal': lambda conf: half_normal(conf['e'], conf['s']),
}


class ParameterValueSampling(object):
    def __init__(self, param_configs):
        """

        :param cparam_configsonfig: A list of dict:
            {param:str, min: number, max: number, distribution: str, dtype: 'float' or 'int'}
        """
        self.param_configs = param_configs

    def get_random_values(self):
        results = list()
        for param_config in self.param_configs:
            param = param_config['param']
            dist = param_config['distribution']
            dist_conf = param_config['distribution_conf']
            dtype = param_config.get('dtype')

            value = sampling[dist](dist_conf)
            if dtype == 'float':
                value = float(value)
            elif dtype == 'int':
                value = int(value)

            results.append((param, value))
        return results


@click.command()
@click.argument('config_file')
def run(config_file):
    """This program is the starting point for every experiment. It pulls together the configuration and all necessary
    experiment classes to load

    """
    config = load_config(config_file)
    config_global = config['global']
    config_random_search = config['random_search']
    logger = setup_logger(config['logger'], 'random_search')

    parameter_value_sampling = ParameterValueSampling(config_random_search['parameters'])

    data_module = config['data-module']
    model_module = config['model-module']
    training_module = config['training-module']
    evaluation_module = config.get('evaluation-module', None)

    DataClass = importlib.import_module(data_module).component
    ModelClass = importlib.import_module(model_module).component
    TrainingClass = importlib.import_module(training_module).component
    EvaluationClass = importlib.import_module(evaluation_module).component

    logger.info('Setting up the data')
    data = DataClass(config['data'], config_global, logger)
    data.setup()

    logger.info('Now starting the random search')
    stop_time = datetime.datetime.now() + datetime.timedelta(hours=config_random_search['max_hours'])
    logger.info('Stopping at {}'.format(stop_time))

    if not os.path.exists(config_random_search['output_path']):
        os.mkdir(config_random_search['output_path'])

    n_runs = 0
    best_run_id, best_run_score, best_run_params = 0, 0, None
    while datetime.datetime.now() < stop_time:
        n_runs += 1
        run_path = os.path.join(config_random_search['output_path'], 'run-{}'.format(n_runs))

        # we build a new config according to the random search configuration
        random_params = parameter_value_sampling.get_random_values()

        logger.info('-' * 20)
        logger.info('Run {}'.format(n_runs))
        logger.info('With params: {}'.format(json.dumps(random_params)))

        config_run = replace_dict_values(config, random_params + [('training.remove_save_folder_on_start', True)])

        with open('{}.yaml'.format(run_path), 'w') as f:
            yaml.dump(config_run, f, default_flow_style=False)

        # we add a new handler to the logger to print all messages to a separate file
        run_log_handler = logging.FileHandler('{}.log'.format(run_path))
        run_log_handler.setLevel(config['logger']['level'])
        run_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(run_log_handler)

        tf.reset_default_graph()
        with tf.Session(config=sess_config) as sess:
            model = ModelClass(config_run['model'], config_global, logger)
            training = TrainingClass(config_run['training'], config_global, logger)
            evaluation = EvaluationClass(config_run['evaluation'], config_global, logger)

            # build the model (e.g. compile it)
            logger.info('Building the model')
            model.build(data, sess)
            # start the training process
            logger.info('Starting the training process')
            best_epoch, best_score = training.start(model, data, sess, evaluation)
            logger.info('DONE')

            # We write the dev score to a file
            with open('{}.score'.format(run_path), 'w') as f:
                f.write('Score: {}\n'.format(best_score))
                f.write('In epoch: {}\n'.format(best_epoch))
                f.write('With params: {}\n'.format(json.dumps(random_params)))

        if best_run_score < best_score:
            best_run_id = n_runs
            best_run_score = best_score
            best_run_params = random_params

        logger.removeHandler(run_log_handler)

    logger.info('-' * 20)
    logger.info('Now stopping. Did perform {} runs.'.format(n_runs))
    logger.info('Best run: id={} with score {}'.format(best_run_id, best_run_score))
    logger.info('Parameters={}'.format(json.dumps(best_run_params)))


if __name__ == '__main__':
    run()
