import tensorflow as tf

from experiment.qa_hinge.model.lw_paper import cosine_similarity, hinge_loss, QAModel


class EmbeddingsModel(QAModel):
    def build(self, data, sess):
        self.build_input(data, sess)

        pooled_question = tf.reduce_mean(self.embeddings_question, [1], keep_dims=False)
        pooled_answer_good = tf.reduce_mean(self.embeddings_answer_good, [1], keep_dims=False)
        pooled_answer_bad = tf.reduce_mean(self.embeddings_answer_bad, [1], keep_dims=False)

        self.question_pooling_weight = 0.0 * tf.to_float(self.input_question)
        self.answer_good_pooling_weight = 0.0 * tf.to_float(self.input_answer_good)
        self.answer_bad_pooling_weight = 0.0 * tf.to_float(self.input_answer_bad)

        self.create_outputs(
            pooled_question,
            pooled_answer_good,
            pooled_question,
            pooled_answer_bad
        )


component = EmbeddingsModel
