from __future__ import division

import math

import numpy as np
import os
from unidecode import unidecode

import experiment


class QAEvaluation(experiment.Evaluation):
    def __init__(self, config, config_global, logger):
        super(QAEvaluation, self).__init__(config, config_global, logger)
        self.length_question = self.config_global['question_length']
        self.length_answer = self.config_global['answer_length']
        self.batchsize_test = self.config.get('batchsize_test', 128)
        self.primary_measure = self.config.get('primary_measure', 'accuracy')

    def start(self, model, data, sess, valid_only=False):
        evaluation_data = [data.archive.valid]
        if not valid_only:
            evaluation_data += data.archive.test

        output_path = self.config.get('output', None)
        if output_path and not os.path.exists(output_path):
            os.mkdir(output_path)

        results = dict()
        for split in evaluation_data:
            self.logger.info("Evaluating {}".format(split.split_name))
            ranks = []
            average_precisions = []

            split_output_path = os.path.join(output_path, split.split_name) if output_path else None
            if split_output_path and not os.path.exists(split_output_path):
                os.mkdir(split_output_path)

            bar = self.create_progress_bar()
            for pool in bar(split.qa):
                test_question = np.array([data.get_item_vector(pool.question, self.length_question)])
                test_answers = np.array([
                    [data.get_item_vector(a, self.length_answer) for a in pool.pooled_answers]
                ])

                scores, = sess.run([model.predict], feed_dict={
                    model.input_question: test_question,
                    model.input_answers: test_answers,
                    model.dropout_keep_prob: 1.0,
                })

                sorted_answers = sorted(zip(scores[0], pool.pooled_answers), key=lambda x: -x[0])
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

                if split_output_path:
                    with open(os.path.join(
                            split_output_path, 'question_{}.txt'.format(pool.question.metadata['id'])
                    ), 'w') as f:
                        f.write(u'Rank: {}\n'.format(rank))
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


component = QAEvaluation
