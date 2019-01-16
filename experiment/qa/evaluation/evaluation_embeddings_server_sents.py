from __future__ import division

import json
from math import ceil

import numpy as np
import requests
import spacy

import experiment


def retrieve_embs(items, server_url, embedding_types, spacy_model, batchsize_embserver, split_sents=False):
    if split_sents:
        sentences = [[s.string for s in spacy_model(q.text, parse=True).sents] for q in items]
    else:
        sentences = [[q.text] for q in items]

    sentences_flat = [s for sub in sentences for s in sub]
    vectors_flat = []

    print('Retrieving {} sents for {} items'.format(len(sentences_flat), len(items)))

    for i in range(int(ceil(len(sentences_flat) / float(batchsize_embserver)))):
        print('batch {} ({}/{} already processed)'.format(i, i * batchsize_embserver, len(sentences_flat)))
        batch_items = sentences_flat[i * batchsize_embserver:(i + 1) * batchsize_embserver]
        config = {'sentences': batch_items, 'embedding_types': embedding_types}
        sentvec_strings = requests.post('{}/embed'.format(server_url), data={'conversion': json.dumps(config)}).text

        embs = np.array([np.fromstring(s, dtype=float, sep=' ') for s in sentvec_strings.split('\n')[:-1]])

        # normalize for later use with cosine similarity via dot product
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True, ord=2)

        embs_l = embs.tolist()
        assert len(embs_l) == len(batch_items)
        vectors_flat += embs.tolist()

    sents_i = 0
    for i, item in enumerate(items):
        ls = len(sentences[i])
        vecs = vectors_flat[sents_i:sents_i+ls]
        assert ls > 0
        assert len(vecs) == ls
        item.metadata['sentence_embeddings'] = vecs
        sents_i += len(sentences[i])


class QAEvaluationEmbeddingsServer(experiment.Evaluation):
    def __init__(self, config, config_global, logger):
        super(QAEvaluationEmbeddingsServer, self).__init__(config, config_global, logger)

        self.server_url = self.config['server_url']
        self.embedding_types = self.config.get('embedding_types', [('glove', ['mean'])])
        self.spacy_model = spacy.load('en')

    def start(self, model, data, sess, valid_only=False):
        if 'sentence_embedding' not in data.archive.questions[0].metadata:
            self.logger.info('Retrieving sentence embeddings for all text items')
            retrieve_embs(data.archive.answers, self.server_url, self.embedding_types, self.spacy_model, self.config['batchsize'], split_sents=True)
            retrieve_embs(data.archive.questions, self.server_url, self.embedding_types, self.spacy_model, self.config['batchsize'], split_sents=False)
            self.logger.info('Got all embeddings!')

        self.score(data.archive.valid)
        for test in data.archive.test:
            self.score(test)

    def score(self, data):
        ranks = []
        average_precisions = []
        for qa in data.qa:
            scores = []
            q_vec = qa.question.metadata['sentence_embeddings'][0]
            for a in qa.pooled_answers:
                score = -100
                for a_vec in a.metadata['sentence_embeddings']:
                    score = max(score, np.dot(q_vec, a_vec))
                scores.append(score)

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
