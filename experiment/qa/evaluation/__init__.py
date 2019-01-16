from __future__ import division

import os
from math import ceil

import numpy as np
from nltk import flatten
from unidecode import unidecode

import experiment



class BasicQAEvaluation(experiment.Evaluation):
    def __init__(self, config, config_global, logger):
        super(BasicQAEvaluation, self).__init__(config, config_global, logger)
        self.primary_measure = self.config.get('primary_measure', 'accuracy')
        self.batchsize = self.config.get('batchsize', 512)

    def start(self, model, data, sess, valid_only=False):
        evaluation_data = [(data.archive.name, data.archive.valid)]
        if not valid_only:
            evaluation_data += [(data.archive.name, t) for t in data.archive.test]
            # transfer datasets
            evaluation_data += [(trans.name, t) for trans in data.transfer_archives for t in trans.test]

        output_path = self.config.get('output', None)
        if output_path and not os.path.exists(output_path):
            os.mkdir(output_path)

        results = dict()
        for dataset_name, split in evaluation_data:
            self.logger.info("Evaluating {} / {}".format(dataset_name, split.split_name))
            ranks = []
            average_precisions = []

            # perform the scoring, used to calculate the measure values
            qa_pairs = [(qa.question, a) for qa in split.qa for a in qa.pooled_answers]
            n_batches = int(ceil(len(qa_pairs) / float(self.batchsize)))
            scores = []
            bar = self.create_progress_bar()
            for i in bar(range(n_batches)):
                batch_qa_pairs = qa_pairs[i * self.batchsize:(i + 1) * self.batchsize]
                result = self.score(batch_qa_pairs, model, data, sess)
                scores += result.tolist()

            scores_used = 0
            for pool in split.qa:
                scores_pool = scores[scores_used:scores_used + len(pool.pooled_answers)]
                scores_used += len(pool.pooled_answers)

                sorted_answers = sorted(zip(scores_pool, pool.pooled_answers), key=lambda x: -x[0])

                rank = 0
                precisions = []
                for i, (score, answer) in enumerate(sorted_answers, start=1):
                    if answer in pool.ground_truth:
                        if rank == 0:
                            rank = i
                        precisions.append((len(precisions) + 1) / float(i))

                ranks.append(rank)
                average_precisions.append(np.mean(precisions))

                if not valid_only:
                    self.logger.debug('Rank: {}'.format(rank))

                # output writing

                split_output_path = os.path.join(output_path, split.split_name) if output_path else None
                if split_output_path and not os.path.exists(split_output_path):
                    os.mkdir(split_output_path)

                if split_output_path:
                    question_path = os.path.join(
                        split_output_path, 'question_{}.txt'.format(pool.question.metadata['id'])
                    )
                    with open(question_path, 'w') as f:
                        f.write(u'Rank: {}\n'.format(rank))
                        f.write(u'Answer Ranking: {}\n'.format(' '.join([a.metadata['id'] for (_, a) in sorted_answers])))
                        try:
                            f.write(u'{}\n'.format(unidecode(pool.question.text)))
                        except:
                            f.write(u'error writing question\n')

                        f.write(u'\n\nGround-Truth:\n' + '-' * 30)
                        for gta in pool.ground_truth:
                            try:
                                filtered_rs = [(i, score) for (i, (score, a)) in enumerate(sorted_answers, start=1) if
                                               a == gta]
                                if filtered_rs:
                                    rank, score = filtered_rs[0]
                                    f.write(u'\n\n{}:{}\n{}\n'.format(rank, score, unidecode(gta.text[:3000])))
                            except:
                                f.write(u'\n\nerror writing answer\n')

                        f.write(u'\n\nTop-10:\n' + '-' * 30)
                        for i, (score, answer) in enumerate(sorted_answers[:10], start=1):
                            try:
                                f.write(u'\n\n{}:{}\n{}\n'.format(i, score, unidecode(answer.text[:3000])))
                            except:
                                f.write(u'\n\n{}\nerror writing answer\n'.format(score))

            correct_answers = len([a for a in ranks if a == 1])
            measures = {
                'accuracy': correct_answers / float(len(ranks)),
                'mrr': np.mean([1 / float(r) for r in ranks]),
                'map': np.mean(average_precisions)
            }

            results[split.split_name] = measures[self.primary_measure]

            self.logger.info('Correct answers: {}/{}'.format(correct_answers, len(split.qa)))
            self.logger.info('Accuracy: {}'.format(measures['accuracy']))
            self.logger.info('MRR: {}'.format(measures['mrr']))
            self.logger.info('MAP: {}'.format(measures['map']))

        return results

    def score(self, question_answer_pairs, model, data, sess):
        raise NotImplementedError()


component = BasicQAEvaluation
