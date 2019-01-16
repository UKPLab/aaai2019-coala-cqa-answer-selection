import tensorflow as tf

from experiment.qa_hinge.model import weight_variable
from experiment.qa_hinge.model.lw_paper.helper.pooling_helper import maxpool, non_zero_tokens, attention_softmax
from experiment.qa_hinge.model.lw_paper.lstm import BiLSTMModel


class AttentionBiLSTMModel(BiLSTMModel):
    def __init__(self, config, config_global, logger):
        super(AttentionBiLSTMModel, self).__init__(config, config_global, logger)
        self.lstm_cell_size = self.config['lstm_cell_size']

    def build(self, data, sess):
        self.build_input(data, sess)

        # we initialize the weights of the representation layers globally so that they can be applied to both, questions
        # and (good/bad)answers. This is an important part, otherwise results would be much worse.
        self.initialize_weights()

        representation_question = maxpool(
            self.bilstm_representation_raw(
                self.embeddings_question,
                self.input_question,
                re_use_lstm=False
            )
        )

        representation_raw_answer_good = self.bilstm_representation_raw(
            self.embeddings_answer_good,
            self.input_answer_good,
            re_use_lstm=True
        )
        representation_raw_answer_bad = self.bilstm_representation_raw(
            self.embeddings_answer_bad,
            self.input_answer_bad,
            re_use_lstm=True
        )

        self.answer_good_pooling_weight = attention_weights(
            representation_raw_answer_good, representation_question, self.W_am, self.W_qm, self.w_ms,
            self.input_answer_good
        )
        self.answer_bad_pooling_weight = attention_weights(
            representation_raw_answer_bad, representation_question, self.W_am, self.W_qm, self.w_ms,
            self.input_answer_bad
        )

        self.question_importance_weight = tf.multiply(self.input_question, 0)
        self.answer_importance_weight = self.answer_good_pooling_weight

        representation_answer_good = maxpool(
            representation_raw_answer_good * tf.expand_dims(self.answer_good_pooling_weight, -1)
        )
        representation_answer_bad = maxpool(
            representation_raw_answer_bad * tf.expand_dims(self.answer_bad_pooling_weight, -1)
        )

        self.create_outputs(
            representation_question,
            representation_answer_good,
            representation_question,
            representation_answer_bad
        )

    def initialize_weights(self):
        """Global initialization of weights for the representation layer

        """
        super(AttentionBiLSTMModel, self).initialize_weights()

        self.W_am = weight_variable('W_am', [self.lstm_cell_size * 2, self.lstm_cell_size * 2])
        self.W_qm = weight_variable('W_qm', [self.lstm_cell_size * 2, self.lstm_cell_size * 2])
        self.w_ms = weight_variable('w_ms', [1, self.lstm_cell_size * 2])


def attention_weights(answer_raw, question, W_am, W_qm, w_ms, answer_indices):
    """This is the attention introduced by Tan et al., ACL 2017

    For each time step in answer_raw we calculate and multiply a weight.
    """
    answer_indices_non_zero = non_zero_tokens(tf.to_float(answer_indices))

    answers_raw_flat = tf.reshape(answer_raw, [-1, tf.shape(answer_raw)[2]])
    W_am_h_flat = tf.transpose(tf.matmul(W_am, tf.transpose(answers_raw_flat, [1, 0])), [1, 0])
    W_am_h = tf.reshape(W_am_h_flat, tf.shape(answer_raw))

    # W_qm_oq
    W_qm_oq = tf.transpose(tf.matmul(W_qm, tf.transpose(question, [1, 0])), [1, 0])

    # m_aq
    m_aq = W_am_h + tf.reshape(W_qm_oq, [-1, 1, tf.shape(answer_raw)[2]])

    # s_aq
    m_aq_tanh = tf.nn.tanh(m_aq)
    m_aq_tanh_flat = tf.reshape(m_aq_tanh, [-1, tf.shape(m_aq_tanh)[2]])
    s_aq_flat = tf.transpose(tf.matmul(w_ms, tf.transpose(m_aq_tanh_flat, [1, 0])), [1, 0])
    s_aq = attention_softmax(tf.reshape(s_aq_flat, [-1, tf.shape(m_aq)[1]]), answer_indices_non_zero)

    return s_aq


component = AttentionBiLSTMModel
