from __future__ import division

import numpy as np

from experiment.qa.evaluation import BasicQAEvaluation


class QAEvaluationPairwise(BasicQAEvaluation):
    def __init__(self, config, config_global, logger):
        super(QAEvaluationPairwise, self).__init__(config, config_global, logger)
        self.length_question = self.config_global['question_length']
        self.length_answer = self.config_global['answer_length']
        self.batchsize_test = self.config.get('batchsize_test', 128)

    def score(self, qa_pairs, model, data, sess):
        questions, answers = zip(*qa_pairs)
        test_questions = np.array([data.get_item_vector(q, self.length_question) for q in questions])
        test_answers = np.array([data.get_item_vector(a, self.length_answer) for a in answers])

        scores, = sess.run([model.predict], feed_dict={
            model.input_question: test_questions,
            model.input_answer: test_answers,
            model.dropout_keep_prob: 1.0,
        })

        return scores


component = QAEvaluationPairwise
