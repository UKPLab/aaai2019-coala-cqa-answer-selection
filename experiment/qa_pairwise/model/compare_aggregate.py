import tensorflow as tf

from experiment.qa_pairwise.model import QAPairwiseModel, bias_variable, weight_variable, multiply_3_2, non_zero_tokens


class CompareAggregateModel(QAPairwiseModel):
    def __init__(self, config, config_global, logger):
        super(CompareAggregateModel, self).__init__(config, config_global, logger)
        self.n_filters = self.config['filters']
        self.preprocess_hidden_size = self.config['preprocess_hidden_size']
        self.classification_hidden_size = self.config.get('classification_hidden_size', self.preprocess_hidden_size)
        self.window_sizes = self.config['window_sizes']
        self.normalization = self.config.get('normalization', True)

    def preprocess(self, item, item_len):
        """Preprocesses an item (Q/A). These are batched!

        :param item:
        :param item_len:
        :return:
        """
        importance = tf.nn.sigmoid(
            tf.nn.bias_add(
                multiply_3_2(item, self.W_i, item_len, self.embedding_size, self.preprocess_hidden_size),
                self.b_i
            )
        )
        transformation = tf.nn.tanh(
            tf.nn.bias_add(
                multiply_3_2(item, self.W_u, item_len, self.embedding_size, self.preprocess_hidden_size),
                self.b_u
            )
        )

        if self.normalization:
            transformation = tf.nn.l2_normalize(transformation, dim=2)

        return tf.multiply(importance, transformation)

    def attention(self, question, answer, non_zero_question_tokens, non_zero_answers_tokens):
        WQ = tf.nn.bias_add(
            multiply_3_2(
                question, self.W_g, self.question_length, self.preprocess_hidden_size, self.preprocess_hidden_size
            ),
            self.b_g
        )

        G = attention_softmax_3d(
            tf.matmul(
                answer,
                WQ,
                transpose_b=True
            ),
            non_zero_question_tokens
        )
        # we set all items to zero that correspond to zero-padded positions of the answer
        G_zero = tf.multiply(G, tf.expand_dims(non_zero_answers_tokens, -1))

        result = tf.matmul(G_zero, question)
        return result

    def classification_layer(self, R):
        """

        :param R: |answers| x filters
        :return:
        """
        dense = tf.nn.tanh(tf.nn.xw_plus_b(R, self.W_2, self.b_2))
        predict = tf.nn.xw_plus_b(dense, tf.expand_dims(self.W_3, -1), self.b_3)
        return tf.reshape(predict, [-1])

    def build(self, data, sess):
        self.build_input(data, sess)
        self.initialize_weights()

        non_zero_question_tokens = non_zero_tokens(tf.to_float(self.input_question))
        non_zero_answer_tokens = non_zero_tokens(tf.to_float(self.input_answer))

        # ~~ Preprocessing
        question = self.preprocess(self.embeddings_question, self.question_length)
        answer = self.preprocess(self.embeddings_answer, self.answer_length)

        # ~~ Attention and Comparison
        H = self.attention(question, answer, non_zero_question_tokens, non_zero_answer_tokens)
        T = tf.multiply(answer, H)

        # ~~ Aggregation
        convolutions = []
        for size in self.window_sizes:
            convoluted = tf.nn.bias_add(
                tf.nn.conv1d(
                    T,
                    self.W_windows[size],
                    stride=1,
                    padding='SAME'
                ),
                self.b_windows[size]
            )
            convolutions.append(convoluted)

        all_convoluted = tf.concat(convolutions, axis=2)
        all_convoluted_padded = tf.multiply(
            all_convoluted,
            tf.tile(
                tf.expand_dims(non_zero_answer_tokens, -1),
                [1, 1, self.n_filters * len(self.window_sizes)]
            )
        )
        R = maxpool(tf.nn.relu(all_convoluted_padded))

        # ~~ Classification
        predict = self.classification_layer(R)

        self.create_outputs(predict)

    def initialize_weights(self):
        """Global initialization of weights for the representation layer

        """

        # preprocessing
        self.W_i = weight_variable('W_i', [self.embedding_size, self.preprocess_hidden_size], reg=self.regularization)
        self.b_i = bias_variable('b_i', [self.preprocess_hidden_size])
        self.W_u = weight_variable('W_u', [self.embedding_size, self.preprocess_hidden_size], reg=self.regularization)
        self.b_u = bias_variable('b_u', [self.preprocess_hidden_size])

        # attention
        self.W_g = weight_variable('W_g', [self.preprocess_hidden_size, self.preprocess_hidden_size], reg=self.regularization)
        self.b_g = bias_variable('b_g', [self.preprocess_hidden_size])

        # aggregation
        self.W_windows = dict()
        self.b_windows = dict()
        for size in self.window_sizes:
            self.W_windows[size] = weight_variable('W_conv_{}'.format(size),
                                                   [size, self.preprocess_hidden_size, self.n_filters], reg=self.regularization)
            self.b_windows[size] = bias_variable('b_conv_{}'.format(size), [self.n_filters])

        # classification
        self.W_2 = weight_variable('W_dense_1', [self.n_filters * len(self.window_sizes), self.classification_hidden_size], reg=self.regularization)
        self.b_2 = bias_variable('b_dense_1', [self.classification_hidden_size])

        self.W_3 = weight_variable('W_dense_out', [self.classification_hidden_size], reg=self.regularization)
        self.b_3 = bias_variable('b_dense_out', [1])


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


component = CompareAggregateModel
