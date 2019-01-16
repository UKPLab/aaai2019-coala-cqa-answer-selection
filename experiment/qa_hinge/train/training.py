import math

import numpy as np

from experiment.qa_hinge.train import QABatchedTrainingHinge


class QATraining(QABatchedTrainingHinge):
    def __init__(self, config, config_global, logger):
        super(QATraining, self).__init__(config, config_global, logger)
        self.n_train_answers = config['n_train_answers']
        self.batchsize_valid = self.config.get('batchsize_valid', 256)

        self.negative_sampling = self.config.get('negative_sampling', 'train')

        self.questions, self.correct_answers, = [], []

        self.batch_i = 0
        self.epoch_random_indices = []

    def prepare_next_epoch(self, model, data, sess, epoch):
        """Prepares the next epoch, especially the batches"""
        super(QATraining, self).prepare_next_epoch(model, data, sess, epoch)

        if len(self.questions) == 0:
            for pool in data.archive.train.qa:
                for gt in pool.ground_truth:
                    self.questions.append(data.get_item_vector(pool.question, self.length_question))
                    self.correct_answers.append(data.get_item_vector(gt, self.length_answer))

        self.batch_i = 0

        # shuffle the indices of each batch
        self.epoch_random_indices = np.random.permutation(len(self.questions))

        # create a list of random answers that is used to sample the incorrect answers

        # create a list of random answers that is used to sample the incorrect answers
        negative_samples = list(data.archive.train.answers)
        if self.negative_sampling == 'all':
            negative_samples += data.archive.additional_answers
        elif self.negative_sampling == 'additional-only':
            negative_samples = list(data.archive.additional_answers)

        n_repetitions = int(math.ceil(
            len(self.questions) / float(len(negative_samples)) * self.n_train_answers
        ))
        repeated_incorrect_answers = list(negative_samples) * n_repetitions
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

        batch_questions, batch_answers_good, batch_answers_bad = [], [], []

        prediction_questions, prediction_answers_bad = [], []

        for i in indices:
            batch_questions.append(self.questions[i])
            batch_answers_good.append(self.correct_answers[i])

            prediction_questions += [self.questions[i]] * self.n_train_answers
            prediction_answers_bad += [data.get_item_vector(next(self.incorrect_answers), self.length_answer)
                                       for _ in range(self.n_train_answers)]

        prediction_results = []
        for predict_batch in range(int(math.ceil(len(prediction_questions) / float(self.batchsize_valid)))):
            batch_start_idx = predict_batch * self.batchsize_valid
            predict_batch_questions = prediction_questions[batch_start_idx: batch_start_idx + self.batchsize_valid]
            predict_batch_answers = prediction_answers_bad[batch_start_idx: batch_start_idx + self.batchsize_valid]

            predictions, = sess.run([model.predict], feed_dict={
                model.input_question: predict_batch_questions,
                model.input_answer_good: predict_batch_answers,
                model.dropout_keep_prob: 1.0
            })
            prediction_results += list(predictions)

        for count, i in enumerate(indices):
            predictions = prediction_results[self.n_train_answers * count:self.n_train_answers * (count + 1)]
            incorrect_answer = prediction_answers_bad[np.argmax(predictions) + self.n_train_answers * count]
            batch_answers_bad.append(incorrect_answer)

        self.batch_i += 1
        return batch_questions, batch_answers_good, batch_answers_bad


component = QATraining
