from __future__ import division

import math
import os

import numpy as np
from unidecode import unidecode

import experiment


class ChooseFirstEvaluation(experiment.Evaluation):
    def __init__(self, config, config_global, logger):
        super(ChooseFirstEvaluation, self).__init__(config, config_global, logger)

    def start(self, model, data, sess):

        evaluation_data = data.archive.test
        if self.config.get('include_valid', False):
            evaluation_data = [data.archive.valid] + evaluation_data

        for split in evaluation_data:
            self.logger.info("Evaluating {}".format(split.split_name))
            ranks = []
            average_precisions = []

            for pool in split.qa:
                sorted_answers = pool.pooled_answers
                rank = 0
                precisions = []
                for i, answer in enumerate(sorted_answers, start=1):
                    if answer in pool.ground_truth:
                        if rank == 0:
                            rank = i
                        precisions.append((len(precisions) + 1) / float(i))

                ranks.append(rank)
                average_precisions.append(np.mean(precisions))
                self.logger.debug('Rank: {}'.format(rank))

            correct_answers = len([a for a in ranks if a == 1])
            accuracy = correct_answers / float(len(ranks))
            mrr = np.mean([1 / float(r) for r in ranks])
            map = np.mean(average_precisions)

            self.logger.info('Correct answers: {}/{}'.format(correct_answers, len(split.qa)))
            self.logger.info('Accuracy: {}'.format(accuracy))
            self.logger.info('MRR: {}'.format(mrr))
            self.logger.info('MAP: {}'.format(map))


component = ChooseFirstEvaluation
