from experiment.util import unique_items


class MetadataItem(object):
    def __init__(self):
        self.metadata = dict()


class Token(MetadataItem):
    def __init__(self, text):
        """

        :type text: str
        """
        super(Token, self).__init__()
        self.text = text


class TextItem(MetadataItem):
    def __init__(self, text, tokens):
        """

        :type text: str
        :type tokens: list[Token]
        """
        super(TextItem, self).__init__()
        self.text = text
        self.tokens = tokens

    @property
    def vocab(self):
        return unique_items([t.text for t in self.tokens])


class QAPool(object):
    def __init__(self, question, pooled_answers, ground_truth):
        """

        :type question: TextItem
        :type pooled_answers: list[TextItem]
        :type ground_truth: list[TextItem]
        """
        self.question = question
        self.pooled_answers = pooled_answers
        self.ground_truth = ground_truth


class Data(object):
    def __init__(self, split_name, qa, answers):
        """

        :type split_name: str
        :type qa: list[QAPool]
        :type answers: list[TextItem]
        """
        self.split_name = split_name
        self.qa = qa
        self.answers = answers

    def combine(self, other_data):
        split_name = self.split_name
        qa = self.qa + other_data.qa
        answers = self.answers + other_data.answers
        return Data(split_name, qa, answers)

    @property
    def questions(self):
        return [p.question for p in self.qa]

    def shrink(self, max_size):
        self.qa = self.qa[:max_size]
        self.answers = unique_items([a for sl in self.qa for a in
                                     (sl.pooled_answers if sl.pooled_answers is not None else sl.ground_truth)])


class Archive(object):
    def __init__(self, name, train, valid, test, questions, answers, additional_answers=None, generated_questions=None):
        """

        :type name: str
        :type train: Data
        :type valid: Data
        :type test: list[Data]
        :type questions: list[TexItem]
        :type answers: list[TexItem]
        """
        self.name = name
        self.train = train
        self.valid = valid
        self.test = test
        self.questions = questions
        self.answers = answers
        self.additional_answers = additional_answers or []
        self.generated_questions = generated_questions or dict()

        self._vocab = None  # lazily created

    def combine(self, other_archive):
        """Creates a combined dataset and returns that

        :param other_archive:
        :return:
        """
        name = '{}+{}'.format(self.name, other_archive.name)
        train = self.train.combine(other_archive.train)
        valid = self.valid.combine(other_archive.valid)
        test = self.test + other_archive.test
        questions = self.questions + other_archive.questions
        answers = self.answers + other_archive.answers
        additional_answers = self.additional_answers + other_archive.additional_answers
        generated_questions = dict(
            list(self.generated_questions.items()) + list(other_archive.generated_questions.items())
        )
        return Archive(name, train, valid, test, questions, answers, additional_answers, generated_questions)

    def shrink(self, split, max_len):
        """Reduces the size of a split of the archive

        :param split:
        :param max_len:
        :return:
        """
        split = {'train': self.train, 'valid': self.valid}[split]
        split.shrink(max_len)

        self.questions = self.train.questions + self.valid.questions + [q for t in self.test for q in t.questions]
        self.answers = unique_items(self.train.answers + self.valid.answers + [a for t in self.test for a in t.answers])

    @property
    def vocab(self):
        """
        :rtype: set
        """
        if self._vocab is None:
            self._vocab = []
            for question in self.questions:
                self._vocab += question.vocab
            for answer in self.answers:
                self._vocab += answer.vocab
            for answer in self.additional_answers:
                self._vocab += answer.vocab
            for questions in self.generated_questions.values():
                for question in questions:
                    self._vocab += question.vocab

            self._vocab = unique_items(self._vocab)

        return self._vocab
