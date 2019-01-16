from __future__ import division

import numpy as np
from nltk import PorterStemmer, OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import experiment


class QAEvaluationTFIDF(experiment.Evaluation):
    def __init__(self, config, config_global, logger):
        super(QAEvaluationTFIDF, self).__init__(config, config_global, logger)
        self.primary_measure = self.config.get('primary_measure', 'accuracy')

    def start(self, model, data, sess, valid_only=False):
        stemmer = PorterStemmer()

        tokens = [' '.join([t.text for t in a.tokens]) for a in data.archive.answers] + \
                 [' '.join([t.text for t in q.tokens]) for q in data.archive.questions]

        tfidf = TfidfVectorizer(tokenizer=lambda x: [stemmer.stem(y) for y in x.split()], stop_words='english')
        # tfidf = TfidfVectorizer(tokenizer=lambda x: [y for y in x.split()], stop_words='english')
        # tfidf = TfidfVectorizer(tokenizer=lambda x: [stemmer.stem(y) for y in x.split()])
        tfidf_matrix = tfidf.fit_transform(tokens)

        results = OrderedDict()
        results[data.archive.valid.split_name] = self.score(data.archive.valid, tfidf_matrix,
                                                            data.archive.answers + data.archive.questions)
        for test in data.archive.test:
            results[test.split_name] = self.score(test, tfidf_matrix, data.archive.answers + data.archive.questions)

        return results

    def score(self, data, tfidf_matrix, all_items):
        self.logger.info("Evaluating {} / {}".format('', data.split_name))

        ranks = []
        average_precisions = []
        for qa in data.qa:
            q_vec = tfidf_matrix[all_items.index(qa.question)]
            a_vecs = [tfidf_matrix[all_items.index(a)] for a in qa.pooled_answers]

            scores = [cosine_similarity(q_vec, a_vec)[0][0] for a_vec in a_vecs]

            sorted_answers = sorted(zip(scores, qa.pooled_answers), key=lambda x: -x[0])
            rank = 0
            precisions = []
            for i, (score, answer) in enumerate(sorted_answers, start=1):
                if answer in qa.ground_truth:
                    if rank == 0:
                        rank = i
                    precisions.append((len(precisions) + 1) / float(i))

            ranks.append(rank)
            average_precisions.append(np.mean(precisions))

            self.logger.debug('Rank: {}'.format(rank))

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

        return measures[self.primary_measure]


component = QAEvaluationTFIDF
