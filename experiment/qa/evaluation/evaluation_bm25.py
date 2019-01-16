from __future__ import division

import numpy as np
from gensim.summarization.bm25 import BM25
from nltk import PorterStemmer

import experiment


class QAEvaluationBM25(experiment.Evaluation):
    def __init__(self, config, config_global, logger):
        super(QAEvaluationBM25, self).__init__(config, config_global, logger)

    def start(self, model, data, sess, valid_only=False):
        stemmer = PorterStemmer()

        texts = []
        for q_a in data.archive.answers + data.archive.questions:
            q_a.metadata['tokens_stemmed'] = [stemmer.stem(t.text) for t in q_a.tokens]
            texts.append(q_a.metadata['tokens_stemmed'])

        bm25 = BM25(texts)
        average_idf = sum(bm25.idf.values()) / float(len(bm25.idf))
        # average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / float(len(bm25.idf.keys()))

        self.score(data.archive.valid, bm25, average_idf, data.archive.answers)
        for test in data.archive.test:
            self.score(test, bm25, average_idf, data.archive.answers)

    def score(self, data, bm25, average_idf, answers):
        ranks = []
        average_precisions = []
        for qa in data.qa:
            scores = bm25.get_scores(qa.question.metadata['tokens_stemmed'], average_idf)
            relevant_scores = [scores[i] for i in [answers.index(a) for a in qa.pooled_answers]]

            sorted_answers = sorted(zip(relevant_scores, qa.pooled_answers), key=lambda x: x[0], reverse=True)
            rank = 0
            precisions = []
            for i, (score, answer) in enumerate(sorted_answers, start=1):
                if answer in qa.ground_truth:
                    if rank == 0:
                        rank = i
                    precisions.append((len(precisions) + 1) / float(i))

            ranks.append(rank)
            average_precisions.append(np.mean(precisions))

        correct_answers = len([a for a in ranks if a == 1])
        measures = {
            'accuracy': correct_answers / float(len(ranks)),
            'mrr': np.mean([1 / float(r) for r in ranks]),
            'map': np.mean(average_precisions)
        }

        self.logger.info('Correct answers: {}/{}'.format(correct_answers, len(data.qa)))
        self.logger.info('Accuracy: {}'.format(measures['accuracy']))
        self.logger.info('MRR: {}'.format(measures['mrr']))
        self.logger.info('MAP: {}'.format(measures['map']))


component = QAEvaluationBM25
