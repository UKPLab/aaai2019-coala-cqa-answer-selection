import tensorflow as tf

from experiment.qa_pairwise.model import QAPairwiseModel, bias_variable, weight_variable


class NewIdeaModel(QAPairwiseModel):
    def __init__(self, config, config_global, logger):
        super(NewIdeaModel, self).__init__(config, config_global, logger)
        self.n_filters = self.config['filters']
        self.window_sizes = self.config.get('window_sizes')
        if not self.window_sizes:
            self.window_sizes = [self.config['window_size']]
        self.normalization = self.config.get('normalization', True)

    def cnn(self, item, window_size):
        self.W_window = weight_variable('W_conv_{}'.format(window_size),
                                   [window_size, self.embedding_size, self.n_filters])
        self.b_window = bias_variable('b_conv_{}'.format(window_size), [self.n_filters])
        return tf.nn.tanh(
            tf.nn.bias_add(
                tf.nn.conv1d(
                    item,
                    self.W_window,
                    stride=1,
                    padding='SAME'
                ),
                self.b_window
            )
        )

    def similarity(self, question_aspects, answer_aspects):
        return tf.matmul(question_aspects, answer_aspects, transpose_b=True)

    def get_question_aspect_coverage(self):
        question = self.embeddings_question
        answer = self.embeddings_answer
        if self.normalization:
            question = tf.nn.l2_normalize(question, dim=2)
            answer = tf.nn.l2_normalize(answer, dim=2)

        # ~~ Aspects
        with tf.variable_scope("aspects") as scope:
            question_aspects = []
            for window_size in self.window_sizes:
                question_aspects.append(self.cnn(question, window_size))
            question_aspects = tf.concat(question_aspects, axis=-1)

            scope.reuse_variables()

            answer_aspects = []
            for window_size in self.window_sizes:
                answer_aspects.append(self.cnn(answer, window_size))
            answer_aspects = tf.concat(answer_aspects, axis=-1)

        # ~~ Comparison
        H = self.similarity(question_aspects, answer_aspects)

        # now we are zeroing-out all entries that are related to zero-padded positions
        H = H * tf.expand_dims(self.non_zero_question_tokens, -1)
        H = H * tf.reshape(self.non_zero_answer_tokens, [-1, 1, self.answer_length])

        question_aspect_coverage = tf.reduce_max(H, axis=2, keep_dims=False)

        # viz
        self.question_importance_weight = question_aspect_coverage
        self.answer_importance_weight = tf.reduce_max(H, axis=1, keep_dims=False)

        return question_aspect_coverage

    def build(self, data, sess):
        self.build_input(data, sess)

        question_aspect_coverage = self.get_question_aspect_coverage()

        tf.summary.histogram('question_aspect_coverage', question_aspect_coverage)
        # also tested harmonic mean, drops performance 8%
        predict = tf.div(
            tf.reduce_sum(question_aspect_coverage, axis=1, keep_dims=False),
            tf.reduce_sum(self.non_zero_question_tokens, axis=1, keep_dims=False)
        )

        self.create_outputs(predict)


def attention_softmax_3d(attention, indices_non_zero):
    """Softmax that ignores values of zero token indices (zero-padding)

    :param attention:
    :param raw_indices:
    :return:
    """
    ex = tf.multiply(tf.exp(attention), tf.expand_dims(indices_non_zero, 1))
    sum = tf.reduce_sum(ex, [2], keep_dims=True)
    softmax = tf.divide(ex, sum)
    return softmax


def maxpool(item):
    return tf.reduce_max(item, [1], keep_dims=False)


component = NewIdeaModel
