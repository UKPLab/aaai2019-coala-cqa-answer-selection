from __future__ import division

import numpy as np
from nltk import bigrams as get_bigrams
from nltk import PorterStemmer, OrderedDict, bigrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import experiment


def feats(ti, stemmer):
    if 'feats' not in ti.metadata:
        bigrams = set(get_bigrams([stemmer.stem(t.text) for t in ti.tokens]))

        # Bigrams sorted (order-independent)
        bigrams = set([tuple(sorted(list(b))) for b in bigrams])

        ti.metadata['feats'] = bigrams

    return ti.metadata['feats']


class QAEvaluationBigrams(experiment.Evaluation):
    def __init__(self, config, config_global, logger):
        super(QAEvaluationBigrams, self).__init__(config, config_global, logger)
        self.primary_measure = self.config.get('primary_measure', 'accuracy')

    def start(self, model, data, sess, valid_only=False):
        stemmer = PorterStemmer()

        results = OrderedDict()
        results[data.archive.valid.split_name] = self.score(data.archive.valid, stemmer,
                                                            data.archive.answers + data.archive.questions)
        for test in data.archive.test:
            results[test.split_name] = self.score(test, stemmer, data.archive.answers + data.archive.questions)

        return results

    def score(self, data, stemmer, all_items):
        ranks = []
        average_precisions = []
        for qa in data.qa:
            # UNIGRAMS OVERRIDE
            # bigrams = lambda x: x

            q_bigrams = feats(qa.question, stemmer)

            as_bigrams = []
            for a in qa.pooled_answers:
                if 'bigrams' not in a.metadata:
                    a.metadata['bigrams'] = feats(a, stemmer)
                as_bigrams.append(a.metadata['bigrams'])

            scores = [len(q_bigrams & ab) for ab in as_bigrams]

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

        correct_answers = len([a for a in ranks if a == 1])
        measures = {
            'accuracy': correct_answers / float(len(ranks)),
            'mrr': np.mean([1 / float(r) for r in ranks]),
            'map': np.mean(average_precisions)
        }

        self.logger.info('Results {}'.format(data.split_name))
        self.logger.info('Correct answers: {}/{}'.format(correct_answers, len(data.qa)))
        self.logger.info('Accuracy: {}'.format(measures['accuracy']))
        self.logger.info('MRR: {}'.format(measures['mrr']))
        self.logger.info('MAP: {}'.format(measures['map']))

        return measures[self.primary_measure]


component = QAEvaluationBigrams
