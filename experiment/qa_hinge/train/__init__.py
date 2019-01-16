from __future__ import division

import shutil

import numpy as np
import os
import tensorflow as tf

import experiment



class QATraining(experiment.Training):
    def __init__(self, config, config_global, logger):
        super(QATraining, self).__init__(config, config_global, logger)

        self.length_question = self.config_global['question_length']
        self.length_answer = self.config_global['answer_length']
        self.n_epochs = self.config['epochs']
        self.n_epochs_per_run = self.config.get('epochs_per_run', self.n_epochs)
        self.batchsize = self.config['batchsize']
        self.dropout_keep_prob = 1.0 - self.config.get('dropout', 0.0)

        # tensorboard summary writer
        self.global_step = 0
        self.train_writer = None

        # checkpointing and weight restoring
        if self.config.get('remove_save_folder_on_start', False) is True and os.path.isdir(self.config['save_folder']):
            self.logger.info('Removing old save folder')
            shutil.rmtree(self.config['save_folder'])

        self.run_recorded_epochs = 0  # number of recorded epochs in this run
        self.state = QATrainState(self.config.get('save_folder', None), less_is_better=False, logger=self.logger)
        self.early_stopping_patience = self.config.get('early_stopping_patience', self.n_epochs)

    def remove_checkpoints(self):
        self.state.clear()

    def start(self, model, data, sess, evaluation):
        if 'tensorflow_log_dir' in self.config_global:
            self.train_writer = tf.summary.FileWriter(self.config_global['tensorflow_log_dir'], sess.graph)
        return 0, 0

    def add_summary(self, summary):
        if self.train_writer:
            self.train_writer.add_summary(summary, self.global_step)

    def record_epoch(self, sess, score):
        self.run_recorded_epochs += 1
        previous_score = self.state.best_score
        self.state.record(sess, score)

        if previous_score != self.state.best_score:
            self.logger.info('Validation score improved from {:.6f} to {:.6f}'.format(
                previous_score, self.state.best_score
            ))
        else:
            self.logger.info('Validation score did not improve ({:.6f}; best={:.6f})'.format(
                score, self.state.best_score
            ))

    def restore_best_weights(self, sess):
        self.state.load(sess, weights='best')

    def is_early_stopping(self):
        return self.state.recorded_epochs - self.state.best_epoch > self.early_stopping_patience


class QABatchedTrainingHinge(QATraining):
    """This is a simple training method that runs over the training data in a linear fashion, just like in keras."""

    def __init__(self, config, config_global, logger):
        super(QABatchedTrainingHinge, self).__init__(config, config_global, logger)
        self.initial_learning_rate = self.config.get('initial_learning_rate', 1.1)
        self.dynamic_learning_rate = self.config.get('dynamic_learning_rate', True)
        self.gradclip = self.config.get('gradclip', False)

        self.epoch_learning_rate = self.initial_learning_rate

    def start(self, model, data, sess, evaluation):
        super(QABatchedTrainingHinge, self).start(model, data, sess, evaluation)

        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer_name = self.config.get('optimizer', 'sgd')
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer_name == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9)
        elif optimizer_name == 'adamax':
            optimizer = AdamaxOptimizer(learning_rate, 0.9, 0.999)
        else:
            raise Exception('No such optimizer: {}'.format(optimizer_name))

        if self.gradclip:
            gradients, variables = zip(*optimizer.compute_gradients(model.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 15.0)
            train = optimizer.apply_gradients(zip(gradients, variables))
        else:
            train = optimizer.minimize(model.loss)

        self.logger.info('Initializing all variables')
        sess.run(tf.global_variables_initializer())

        self.state.load(sess, weights='last')
        start_epoch = self.state.recorded_epochs + 1
        end_epoch = min(self.n_epochs + 1, start_epoch + self.n_epochs_per_run)

        if self.state.recorded_epochs > 0:
            self.logger.info('Loaded the weights of last epoch {} with score={}'.format(
                self.state.recorded_epochs, self.state.scores[-1]
            ))
            if not self.is_early_stopping() and start_epoch < end_epoch:
                self.logger.info('Now calculating validation score (to verify the restoring success)')
                valid_score = list(evaluation.start(model, data, sess, valid_only=True).values())[0]
                self.logger.info('Score={:.4f}'.format(valid_score))

        self.logger.info('Running from epoch {} to epoch {}'.format(start_epoch, end_epoch - 1))

        for epoch in range(start_epoch, end_epoch):
            if self.is_early_stopping():
                self.logger.info('Early stopping (no improvement in the last {} epochs)'.format(
                    self.state.recorded_epochs - self.state.best_epoch
                ))
                break

            self.logger.info('Epoch {}/{}'.format(epoch, self.n_epochs))

            self.logger.debug('Preparing epoch')
            self.prepare_next_epoch(model, data, sess, epoch)

            bar = self.create_progress_bar('loss')
            train_losses = []  # used to calculate the epoch train loss
            recent_train_losses = []  # used to calculate the display loss

            self.logger.debug('Training')
            self.logger.debug('{} minibatches with size {}'.format(self.get_n_batches(), self.batchsize))
            for _ in bar(range(int(self.get_n_batches()))):
                self.global_step += self.batchsize
                train_questions, train_answers_good, train_answers_bad = self.get_next_batch(model, data, sess)

                _, loss, loss_individual, summary = sess.run(
                    [train, model.loss, model.loss_individual, model.summary],
                    feed_dict={
                        learning_rate: self.epoch_learning_rate,
                        model.input_question: train_questions,
                        model.input_answer_good: train_answers_good,
                        model.input_answer_bad: train_answers_bad,
                        model.dropout_keep_prob: self.dropout_keep_prob
                    })
                recent_train_losses = ([loss] + recent_train_losses)[:20]
                train_losses.append(loss)
                bar.dynamic_messages['loss'] = np.mean(recent_train_losses)
                self.add_summary(summary)
            self.logger.info('train loss={:.6f}'.format(np.mean(train_losses)))

            self.logger.info('Now calculating validation score')
            valid_score = list(evaluation.start(model, data, sess, valid_only=True).values())[0]

            # if the validation score is better than the best observed previous loss, create a checkpoint
            self.record_epoch(sess, valid_score)

        return self.state.best_epoch, self.state.best_score

    def restore_best_epoch(self, model, data, sess, evaluation):
        self.logger.info('Restoring the weights of the best epoch {} with score {}'.format(
            self.state.best_epoch, self.state.best_score
        ))
        self.state.load(sess, weights='best')

        self.logger.info('Now calculating validation score (to verify the restoring success)')
        valid_score = list(evaluation.start(model, data, sess, valid_only=True).values())[0]
        self.logger.info('Score={:.4f}'.format(valid_score))

    def prepare_next_epoch(self, model, data, sess, epoch):
        """Prepares the next epoch, especially the batches"""
        self.epoch_learning_rate = self.initial_learning_rate
        if self.dynamic_learning_rate:
            self.epoch_learning_rate /= float(epoch)

    def get_n_batches(self):
        raise NotImplementedError()

    def get_next_batch(self, model, data, sess):
        """Return the training data for the next batch

        :return: questions, answers, labels
        :rtype: list, list, list
        """
        pass


class QATrainState(object):
    def __init__(self, path, less_is_better, logger):
        """Represents the a training state

        :param path: the folder where the checkpoints should be written to
        :param less_is_better: True if a smaller validation score is desired
        """
        self.path = path
        self.logger = logger
        self.less_is_better = less_is_better
        self._saver = None

        self.initialize()

    def initialize(self):
        self.scores = []
        self.best_score = -1 if not self.less_is_better else 2
        self.best_epoch = 0
        self.recorded_epochs = 0
        if self.path and not os.path.exists(self.path):
            os.mkdir(self.path)

    def load(self, session, weights='last'):
        """

        :param session:
        :param weights: 'last' or 'best'
        :return:
        """
        if os.path.exists(self.scores_file):
            scores = []
            with open(self.scores_file, 'r') as f:
                for line in f:
                    scores.append(float(line))

            self.scores = scores
            op = max if not self.less_is_better else min
            self.best_score = op(scores)
            self.best_epoch = scores.index(self.best_score) + 1
            self.recorded_epochs = len(scores)

        restore_path = '{}-{}'.format(
            self.checkpoint_file,
            self.recorded_epochs if weights == 'last' else self.best_epoch
        )
        if os.path.exists(restore_path) or os.path.exists('{}.index'.format(restore_path)):
            self.saver.restore(session, restore_path)
        else:
            self.logger.info('Could not restore weights. Path does not exist: {}'.format(restore_path))

    def record(self, session, score):
        self.recorded_epochs += 1
        self.scores.append(score)
        with open(self.scores_file, 'a') as f:
            f.write('{}\n'.format(score))
        self.saver.save(session, self.checkpoint_file, global_step=self.recorded_epochs)

        if (not self.less_is_better and score > self.best_score) or (self.less_is_better and score < self.best_score):
            self.best_score = score
            self.best_epoch = self.recorded_epochs

    def clear(self):
        shutil.rmtree(self.path)
        self._saver = None
        self.initialize()

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=None)
        return self._saver

    @property
    def scores_file(self):
        return os.path.join(self.path, 'scores.txt')

    @property
    def checkpoint_file(self):
        return os.path.join(self.path, 'model-checkpoint')


from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class AdamaxOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm.
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")