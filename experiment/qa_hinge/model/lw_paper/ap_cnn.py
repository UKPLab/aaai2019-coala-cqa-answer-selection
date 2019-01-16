from experiment.qa_hinge.model import weight_variable
from experiment.qa_hinge.model.lw_paper.cnn import CNNModel
from experiment.qa_hinge.model.lw_paper.helper.pooling_helper import attentive_pooling_weights, weighted_pooling


class AttentivePoolingCNNModel(CNNModel):
    def build(self, data, sess):
        self.build_input(data, sess)
        self.initialize_weights()

        raw_representation_question = self.cnn_representation_raw(self.embeddings_question, self.question_length)
        raw_representation_answer_good = self.cnn_representation_raw(self.embeddings_answer_good, self.answer_length)
        raw_representation_answer_bad = self.cnn_representation_raw(self.embeddings_answer_bad, self.answer_length)

        self.question_good_pooling_weight, self.answer_good_pooling_weight = attentive_pooling_weights(
            self.U_AP,
            raw_representation_question,
            raw_representation_answer_good,
            self.input_question,
            self.input_answer_good
        )
        self.question_bad_pooling_weight, self.answer_bad_pooling_weight = attentive_pooling_weights(
            self.U_AP,
            raw_representation_question,
            raw_representation_answer_bad,
            self.input_question,
            self.input_answer_bad
        )

        self.question_pooling_weight = self.question_good_pooling_weight

        pooled_representation_question_good = weighted_pooling(
            raw_representation_question, self.question_good_pooling_weight, self.input_question
        )
        pooled_representation_answer_good = weighted_pooling(
            raw_representation_answer_good, self.answer_good_pooling_weight, self.input_answer_good
        )
        pooled_representation_question_bad = weighted_pooling(
            raw_representation_question, self.question_bad_pooling_weight, self.input_question
        )
        pooled_representation_answer_bad = weighted_pooling(
            raw_representation_answer_bad, self.answer_bad_pooling_weight, self.input_answer_bad
        )

        self.create_outputs(
            pooled_representation_question_good,
            pooled_representation_answer_good,
            pooled_representation_question_bad,
            pooled_representation_answer_bad
        )

    def initialize_weights(self):
        """Global initialization of weights for the representation layer

        """
        super(AttentivePoolingCNNModel, self).initialize_weights()
        self.U_AP = weight_variable('U_AP', [self.n_filters, self.n_filters])


component = AttentivePoolingCNNModel
