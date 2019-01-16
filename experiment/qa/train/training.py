import math

import numpy as np

from experiment.qa.train import QABatchedTraining


class QATrainingSimple(QABatchedTraining):
    """This is a simple training method that runs over the training data in a linear fashion, just like in keras."""

    def __init__(self, config, config_global, logger):
        super(QATrainingSimple, self).__init__(config, config_global, logger)
        self.n_train_answers = config['n_train_answers']

        self.questions, self.correct_answers, = [], []

        self.batch_i = 0
        self.epoch_random_indices = []

    def prepare_next_epoch(self, model, data, sess, epoch):
        """Prepares the next epoch, especially the batches"""
        super(QATrainingSimple, self).prepare_next_epoch(model, data, sess, epoch)

        if len(self.questions) == 0:
            for pool in data.archive.train.qa:
                for gt in pool.ground_truth:
                    self.questions.append(data.get_item_vector(pool.question, self.length_question))
                    self.correct_answers.append(data.get_item_vector(gt, self.length_answer))

        self.batch_i = 0

        # shuffle the indices of each batch
        self.epoch_random_indices = np.random.permutation(len(self.questions))

        # create a list of random answers that is used to sample the incorrect answers
        n_repititions = int(math.ceil(
            len(self.questions) / float(len(data.archive.train.answers)) * self.n_train_answers
        ))
        repeated_incorrect_answers = list(data.archive.train.answers) * n_repititions
        np.random.shuffle(repeated_incorrect_answers)
        self.incorrect_answers = iter(repeated_incorrect_answers)

    def get_n_batches(self):
        return math.ceil(len(self.questions) / self.batchsize)

    def get_next_batch(self, model, data, sess):
        """Return the training data for the next batch

        :return: questions, good answers, bad answers
        :rtype: list, list, list
        """
        indices = self.epoch_random_indices[self.batch_i * self.batchsize: (self.batch_i + 1) * self.batchsize]

        batch_questions, batch_answers, batch_labels = [], [], []
        for i in indices:
            question = self.questions[i]
            answers = [self.correct_answers[i]] + [
                data.get_item_vector(next(self.incorrect_answers), self.length_answer)
                for _ in range(self.n_train_answers - 1)
            ]
            labels = [1.0] + [0.0] * (self.n_train_answers - 1)
            batch_questions.append(question)
            batch_answers.append(answers)
            batch_labels.append(labels)

        self.batch_i += 1
        return batch_questions, batch_answers, batch_labels


component = QATrainingSimple
