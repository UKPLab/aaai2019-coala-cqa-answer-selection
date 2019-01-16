import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer

from experiment import Model


class QAPairwiseModel(Model):
    def __init__(self, config, config_global, logger):
        super(QAPairwiseModel, self).__init__(config, config_global, logger)
        self.__summary = None

        self.trainable_embeddings = self.config.get('trainable_embeddings', True)
        self.question_length = self.config_global['question_length']
        self.answer_length = self.config_global['answer_length']
        self.embedding_size = self.config_global['embedding_size']
        self.regularization = self.config.get('regularization')
        self.regularization_alpha = self.config.get('regularization_alpha', 0.0)

    def build_input(self, data, sess):
        # First None: Minibatch; Second None: Answers
        self.input_question = tf.placeholder(tf.int32, [None, self.question_length])
        self.input_answer = tf.placeholder(tf.int32, [None, self.answer_length])
        self.input_label = tf.placeholder(tf.float32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        # for reranking server with attention viz
        self.question_importance_weight = tf.multiply(self.input_question, 0)
        self.answer_importance_weight = tf.multiply(self.input_question, 0)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embeddings_init = tf.constant_initializer(data.embeddings)
            embeddings_weight = tf.get_variable("embeddings", data.embeddings.shape, dtype=tf.float32,
                                                initializer=embeddings_init,
                                                trainable=self.trainable_embeddings)

            self.embeddings_question = tf.nn.dropout(
                tf.nn.embedding_lookup(embeddings_weight, self.input_question),
                self.dropout_keep_prob,
            )
            self.embeddings_answer = tf.nn.dropout(
                tf.nn.embedding_lookup(embeddings_weight, self.input_answer),
                self.dropout_keep_prob
            )

        self.non_zero_question_tokens = non_zero_tokens(tf.to_float(self.input_question))
        self.non_zero_answer_tokens = non_zero_tokens(tf.to_float(self.input_answer))

    def create_outputs(self, prediction):
        self.predict = tf.nn.sigmoid(prediction)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = self.regularization_alpha

        self.loss_individual = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label, logits=prediction)
        self.loss = tf.reduce_mean(self.loss_individual) + reg_constant * sum(reg_losses)

        tf.summary.scalar('Loss', self.loss)

    @property
    def summary(self):
        if self.__summary is None:
            self.__summary = tf.summary.merge_all(key='summaries')
        return self.__summary


def weight_variable(name, shape, reg=None):
    if reg is not None:
        reg = tf.contrib.layers.l2_regularizer(scale=reg)
    return tf.get_variable(name, shape=shape, initializer=xavier_initializer(), regularizer=reg)


def bias_variable(name, shape, value=0.0):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(value))


def maxpool(item):
    return tf.reduce_max(item, [1], keep_dims=False)


def multiply_3_2(x, y, n_items=None, n_values=None, n_output_values=None):
    """Matmuls each 2d matrix in a 3d tensor with a 2d mulitplicator

    :param x: 3d input
    :param y: 2d input
    :param n_items: you can explicitly set the shape of the input to enable better debugging in tensorflow
    :return:
    """
    shape_x = tf.shape(x)
    shape_y = tf.shape(y)

    n_items = shape_x[1] if n_items is None else n_items
    n_values = shape_x[2] if n_values is None else n_values
    n_output_values = shape_y[1] if n_output_values is None else n_output_values

    x_2d = tf.reshape(x, [-1, n_values])
    result_2d = tf.matmul(x_2d, y)
    result_3d = tf.reshape(result_2d, [-1, n_items, n_output_values])
    return result_3d


def non_zero_tokens(tokens):
    """Receives a batch of vectors of tokens (float) which are zero-padded. Returns a vector of the same size, which has
    the value 1.0 in positions with actual tokens and 0.0 in positions with zero-padding.

    :param tokens:
    :return:
    """
    return tf.ceil(tokens / tf.reduce_max(tokens, [1], keep_dims=True))

