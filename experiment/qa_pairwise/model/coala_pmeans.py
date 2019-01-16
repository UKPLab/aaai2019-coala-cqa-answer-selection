import numpy as np
import tensorflow as tf
from experiment.qa_pairwise.model.new_idea import NewIdeaModel

from experiment.qa_pairwise.model import bias_variable, weight_variable


class NewIdeaClassificationPMeansModel(NewIdeaModel):
    def __init__(self, config, config_global, logger):
        super(NewIdeaClassificationPMeansModel, self).__init__(config, config_global, logger)
        self.n_p_values = self.config.get('n_p_values', 100)
        self.hidden_size = self.config.get('hidden_size', self.n_p_values)

    def build(self, data, sess):
        self.build_input(data, sess)

        question_aspect_coverage = self.get_question_aspect_coverage()

        # ones are very good, zeros are OK, random is bad
        # init_p_values = np.random.normal(0, 1, self.n_p_values)
        # init_p_values = np.random.normal(1, 0.5, self.n_p_values)
        # init_p_values = np.zeros([self.n_p_values])
        init_p_values = np.ones([self.n_p_values])

        p_values = []
        for i in range(self.n_p_values):
            p = tf.get_variable('p_{}'.format(i), [1], initializer=tf.constant_initializer(init_p_values[i]))
            p_values.append(p)
            self.special_print['p_{}'.format(i)] = p

        p_means = [learned_p_mean(question_aspect_coverage, p_values[i], self.non_zero_question_tokens)
                   for i in range(self.n_p_values)]

        concat = tf.concat(p_means, axis=1)

        W_result_1 = weight_variable('W_result_1', [self.n_p_values, self.hidden_size])
        b_result_1 = bias_variable('b_result_1', [self.hidden_size])
        predict_1 = tf.nn.relu(tf.nn.xw_plus_b(concat, W_result_1, b_result_1))
        W_result_2 = weight_variable('W_result_2', [self.hidden_size, 1])
        b_result_2 = bias_variable('b_result_2', [1])
        predict = tf.reshape(tf.nn.xw_plus_b(predict_1, W_result_2, b_result_2), [-1])

        self.create_outputs(predict)


def p_mean(values, p, non_zero_question_tokens):
    p_tf = tf.constant(float(p), dtype=tf.float32)
    n_toks = tf.reduce_sum(non_zero_question_tokens, keep_dims=True, axis=1)

    values = tf.nn.relu(values) + 0.001  # epsilon
    res = tf.pow(
        tf.reduce_sum(
            tf.pow(values, p_tf) * non_zero_question_tokens,
            axis=1,
            keep_dims=True
        ) / n_toks,
        1.0 / p_tf
    )

    # res = tf.Print(res, [res[0], values[0]])
    return res


p_count = 0


def learned_p_mean(values, p, non_zero_question_tokens):
    epsilon = 0.001

    n_toks = tf.reduce_sum(non_zero_question_tokens, keep_dims=True, axis=1)
    p_nonzero = tf.cond(p[0] >= 0, lambda: p + epsilon, lambda: p - epsilon)

    values = tf.nn.relu(values) + epsilon
    res = tf.pow(
        tf.reduce_sum(
            tf.pow(values, p_nonzero) * non_zero_question_tokens,
            axis=1,
            keep_dims=True
        ) / n_toks,
        1.0 / p_nonzero
    )

    # res = tf.Print(res, [res[0], values[0]])
    return res


component = NewIdeaClassificationPMeansModel
