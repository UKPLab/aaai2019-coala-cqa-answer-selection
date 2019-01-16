import os
from collections import defaultdict

import numpy as np

from experiment.qa.data.models import *
from experiment.qa.data.reader import TSVArchiveReader


class TSVReader(TSVArchiveReader):
    def __init__(self, archive_path, lowercased, logger, generated_questions_path):
        super(TSVReader, self).__init__(archive_path, lowercased, logger)
        self.generated_questions_path = generated_questions_path

    def file_path(self, filename):
        return os.path.join(self.archive_path, filename)

    def read_items(self, name, vocab):
        items_path = self.file_path('{}.tsv.gz'.format(name))
        items = dict()
        for line in self.read_tsv(items_path, is_gzip=True):
            id = line[0]
            text = line[1] if len(line) > 1 else ''
            tokens = [Token(vocab[t]) for t in text.split()]
            answer = TextItem(' '.join(t.text for t in tokens), tokens)
            answer.metadata['id'] = id
            items[id] = answer
        return items

    def read_split(self, name, questions, answers):
        split_path = self.file_path('{}.tsv.gz'.format(name))
        datapoints = []
        split_answers = []
        for i, line in enumerate(self.read_tsv(split_path, is_gzip=True)):
            question = questions[line[0]]
            ground_truth = [answers[gt_id] for gt_id in line[1].split()]
            pool = [answers[pa_id] for pa_id in line[2].split()] if len(line) > 2 else None
            if pool is not None:
                np.random.shuffle(pool)
            datapoints.append(QAPool(question, pool, ground_truth))

            if pool is not None:
                split_answers += pool
            else:
                split_answers += ground_truth

        # we filter out all pools that do not contain any ground truth answer
        if name != 'train':
            # we filter out all pools that do not contain any ground truth answer
            qa_pools_len_before = len(datapoints)
            datapoints = [p for p in datapoints if len([1 for gt in p.ground_truth if gt in p.pooled_answers]) > 0]
            qa_pools_len_after = len(datapoints)
            self.logger.info("Split {} reduced to {} item from {} due to missing ground truth in pool".format(
                name, qa_pools_len_after, qa_pools_len_before
            ))

        return Data(name, datapoints, split_answers)

    def read(self):
        vocab = dict(self.read_tsv(self.file_path('vocab.tsv.gz'), is_gzip=True))

        answers = self.read_items('answers', vocab)
        questions = self.read_items('questions', vocab)
        try:
            additional_answers = self.read_items("additional-answers", vocab)
            additional_answers = list(additional_answers.values())
            self.logger.info('Read {} additional (unrelated) answers'.format(len(additional_answers)))
        except:
            additional_answers = None
            self.logger.info('No additional answers found for this dataset')

        train = self.read_split("train", questions, answers)
        valid = self.read_split("valid", questions, answers)
        test = self.read_split("test", questions, answers)

        self.logger.debug('Average answer length: {}'.format(
            np.mean([len(a.tokens) for a in answers.values()])
        ))

        generated_questions = self.read_generated_questions()
        self.logger.debug('Did load generated questions for {} data points'.format(len(generated_questions.keys())))

        return Archive(
            self.name, train, valid, [test],
            list(questions.values()), list(answers.values()),
            additional_answers, generated_questions
        )

    def read_generated_questions(self):
        result = defaultdict(lambda: list())
        if self.generated_questions_path:
            for post_id, distance, generated_question_text in self.read_tsv(self.generated_questions_path):
                toks = [Token(t) for t in generated_question_text.split(' ')]
                # if len(toks) < 20:
                question = TextItem(generated_question_text, toks)
                question.metadata['distance'] = distance
                result[post_id].append(question)
        return result
