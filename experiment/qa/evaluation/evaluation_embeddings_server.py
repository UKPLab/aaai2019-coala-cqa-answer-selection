from __future__ import division

import json
from math import ceil

import numpy as np
import requests

import experiment


def retrieve_embs(items, server_url, embedding_types):
    batchsize_embserver = 512
    for i in range(int(ceil(len(items) / float(batchsize_embserver)))):
        batch_items = items[i * batchsize_embserver:(i + 1) * batchsize_embserver]
        texts = [q.text for q in batch_items]
        config = {'sentences': texts, 'embedding_types': embedding_types}
        sentvec_strings = requests.post('{}/embed'.format(server_url), data={'conversion': json.dumps(config)}).text

        embs = np.array([np.fromstring(s, dtype=float, sep=' ') for s in sentvec_strings.split('\n')[:-1]])
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True, ord=2)

        for batch_item, emb in zip(batch_items, embs):
            batch_item.metadata['sentence_embedding'] = emb


class QAEvaluationEmbeddingsServer(experiment.Evaluation):
    def __init__(self, config, config_global, logger):
        super(QAEvaluationEmbeddingsServer, self).__init__(config, config_global, logger)

        self.server_url = self.config['server_url']
        self.embedding_types = self.config.get('embedding_types', [('glove', ['mean'])])

    def start(self, model, data, sess, valid_only=False):
        if 'sentence_embedding' not in data.archive.questions[0].metadata:
            self.logger.info('Retrieving sentence embeddings for all text items')
            retrieve_embs(data.archive.questions + data.archive.answers, self.server_url, self.embedding_types)

        self.score(data.archive.valid)
        for test in data.archive.test:
            self.score(test)

    def score(self, data):
        ranks = []
        average_precisions = []
        for qa in data.qa:
            scores = [np.dot(qa.question.metadata['sentence_embedding'], a.metadata['sentence_embedding']) for a in
                      qa.pooled_answers]

            sorted_answers = sorted(zip(scores, qa.pooled_answers), key=lambda x: x[0], reverse=True)
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


component = QAEvaluationEmbeddingsServer
