import numpy as np

import experiment
from experiment.qa.data import models
from experiment.util import read_embeddings, unique_items


class QAData(experiment.Data):
    def __init__(self, config, config_global, logger):
        super(QAData, self).__init__(config, config_global, logger)

        # public fields
        self.archive = None  # Archive
        self.transfer_archives = []  # list[Archive]
        self.vocab_to_index = dict()  # a dict that matches each token to an integer position for the embeddings
        self.embeddings = None  # numpy array that contains all relevant embeddings
        self.lowercased = self.config.get('lowercased', False)

        self.map_oov = self.config.get('map_oov', False)
        self.map_numbers = self.config.get('map_numbers', False)

        self.max_train_samples = self.config.get('max_train_samples')

    def _get_train_readers(self):
        """:rtype: list[ArchiveReader]"""
        raise NotImplementedError()

    def _get_transfer_readers(self):
        """:rtype: list[ArchiveReader]"""
        return []

    def setup(self):
        readers = self._get_train_readers()
        self.logger.info('Loading {} train datasets'.format(len(readers)))
        archives = [reader.read() for reader in readers]
        self.archive = archives[0]

        if self.max_train_samples:
            for archive in archives:
                archive.shrink('train', self.max_train_samples)
                print('')
            self.logger.info('Reduced the maximum of training sample of all data archives to a maximum of {}'.format(
                self.max_train_samples
            ))

        if self.config.get('balance_data') is True:
            max_len_train = min([len(a.train.qa) for a in archives])
            max_len_dev = min([len(a.valid.qa) for a in archives])
            for archive in archives:
                archive.shrink('train', max_len_train)
                archive.shrink('valid', max_len_dev)
                # archive.train.qa = archive.train.qa[:max_len_train]
                # archive.valid.qa = archive.valid.qa[:max_len_dev]

            self.logger.info('Balanced all data archives to maximum length for train={}, dev={}'.format(
                max_len_train, max_len_dev
            ))

        for other_archive in archives[1:]:
            self.archive = self.archive.combine(other_archive)
        self.logger.debug('Train dataset questions: train={}, dev={}, test={}'.format(
            len(self.archive.train.qa),
            len(self.archive.valid.qa),
            [len(t.qa) for t in self.archive.test]
        ))

        qas = self.archive.train.qa + self.archive.valid.qa
        for t in self.archive.test:
            qas += t.qa
        self.logger.debug('Mean answer count={}'.format(
            np.mean([len(p.ground_truth) for p in qas])
        ))
        self.logger.debug('Mean poolsize={}'.format(
            np.mean([len(p.pooled_answers) for p in qas if p.pooled_answers is not None])
        ))

        self.transfer_archives = [r.read() for r in self._get_transfer_readers()]
        if self.transfer_archives:
            self.logger.debug('Transfer datasets with test questions: {}'.format(
                ', '.join(['{}={}'.format(a.name, [len(t.qa) for t in a.test]) for a in self.transfer_archives])
            ))

        if 'embeddings_path' in self.config:
            # load the initial embeddings
            self.logger.info('Fetching the dataset vocab')
            vocab = unique_items(self.archive.vocab + [b for a in self.transfer_archives for b in a.vocab])
            self.logger.info('Loading embeddings (vocab size={})'.format(len(vocab)))

            embeddings_paths = self.config['embeddings_path']
            if isinstance(embeddings_paths, str):
                embeddings_paths = [embeddings_paths]

            embeddings_dicts = [read_embeddings(p, vocab, self.logger) for p in embeddings_paths]
            embeddings_dicts_sizes = [len(next(ed.itervalues())) for ed in embeddings_dicts]
            embedding_size = sum(embeddings_dicts_sizes)

            zero_padding = np.zeros((embedding_size,))
            oov = np.random.uniform(-1.0, 1.0, [embedding_size, ])
            # oov = np.zeros([embedding_size, ])
            embeddings = [zero_padding, oov]

            n_oov = 0
            for token in self.archive.vocab:
                embedding_dict_items = [ed.get(token, None) for ed in embeddings_dicts]
                is_oov = all(v is None for v in embedding_dict_items)
                embedding_dict_items = [
                    x if x is not None else np.random.uniform(-1.0, 1.0, [embeddings_dicts_sizes[i], ])
                    for (i, x) in enumerate(embedding_dict_items)
                ]

                if not is_oov:
                    self.vocab_to_index[token] = len(embeddings)
                    embeddings.append(np.hstack(embedding_dict_items))
                else:
                    n_oov += 1
                    if self.map_oov:
                        self.vocab_to_index[token] = 1  # oov
                    else:
                        # for each oov, we create a new random vector
                        self.vocab_to_index[token] = len(embeddings)
                        embeddings.append(np.random.uniform(-1.0, 1.0, [embedding_size, ]))

            self.embeddings = np.array(embeddings)
            self.logger.info('OOV tokens: {}'.format(n_oov))

        else:
            embedding_size = self.config_global['embedding_size']
            self.vocab_to_index = dict([(t, i) for (i, t) in enumerate(self.archive.vocab, start=1)])
            self.embeddings = np.append(
                np.zeros((1, embedding_size)),  # zero-padding
                np.random.uniform(-1.0, 1.0, [len(self.archive.vocab), embedding_size]),
                axis=0
            )

    def get_item_vector(self, ti, max_len):
        key = 'vec-{}'.format(max_len)
        if key not in ti.metadata:
            tokens = [self.vocab_to_index[t.text] if t.text in self.vocab_to_index else 1 for t in ti.tokens[:max_len]]
            # zero-pad to max_len
            tokens_padded = tokens + [0 for _ in range(max_len - len(tokens))]
            ti.metadata[key] = tokens_padded
        return ti.metadata[key]

    def get_item_vectors_split(self, ti, max_len):
        """This is just like get_item_vector but it splits the text item into two halves and returns

        :param ti:
        :param max_len:
        :return:
        """
        key = 'vec-split-{}'.format(max_len)
        if key not in ti.metadata:
            split_i = int(min(max_len/2, len(ti.tokens)/2))
            toks_1 = ti.tokens[:split_i]
            toks_2 = ti.tokens[split_i:max_len]

            toks_1_index = [self.vocab_to_index[t.text] if t.text in self.vocab_to_index else 1 for t in toks_1]
            toks_2_index = [self.vocab_to_index[t.text] if t.text in self.vocab_to_index else 1 for t in toks_2]
            # zero-pad to max_len
            toks_1_index_padded = toks_1_index + [0 for _ in range(max_len - len(toks_1_index))]
            toks_2_index_padded = toks_2_index + [0 for _ in range(max_len - len(toks_2_index))]
            ti.metadata[key] = toks_1_index_padded, toks_2_index_padded
        return ti.metadata[key]

    def get_items(self, qa, negative_answers=None):
        """Returns randomly constructed samples for all questions inside the qa list with a specific number of negative
        answers

        :param qa:
        :param negative_answers: None=all
        :return:
        """
        questions = []
        answers_good = []
        answers_bad = []

        length_question = self.config_global['question_length']
        length_answer = self.config_global['answer_length']

        for pool in qa:
            question = self.get_item_vector(pool.question, length_question)
            for answer_good_item in pool.ground_truth:
                answer_good = self.get_item_vector(answer_good_item, length_answer)
                shuffled_pool = [pa for pa in pool.pooled_answers if pa not in pool.ground_truth]
                np.random.shuffle(shuffled_pool)
                for answer_bad_item in shuffled_pool[:negative_answers]:
                    answer_bad = self.get_item_vector(answer_bad_item, length_answer)
                    questions.append(question)
                    answers_good.append(answer_good)
                    answers_bad.append(answer_bad)

        questions = np.array(questions, dtype=np.int32)
        answers_good = np.array(answers_good, dtype=np.int32)
        answers_bad = np.array(answers_bad, dtype=np.int32)

        return questions, answers_good, answers_bad
