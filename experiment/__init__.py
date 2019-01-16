from collections import OrderedDict

import progressbar


class ComponentBase(object):
    def __init__(self, config, config_global, logger):
        """This is a simple base object for all experiment components

        :type config: dict
        :type config_global: dict
        :type logger: logging.Logger
        """
        self.config = config or dict()
        self.config_global = config_global or dict()
        self.logger = logger

    def create_progress_bar(self, dynamic_msg=None):
        widgets = [
            ' [batch ', progressbar.SimpleProgress(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') '
        ]
        if dynamic_msg is not None:
            widgets.append(progressbar.DynamicMessage(dynamic_msg))

        if self.config_global.get('show_progress', True):
            return progressbar.ProgressBar(widgets=widgets)
        else:
            return progressbar.NullBar()


class Data(ComponentBase):
    def setup(self):
        pass

    def get_fold_data(self, fold_i, n_folds):
        """Generates and returns a new Data instance that contains only the data for a specific fold. This method is
        used for hyperparameter optimization on multiple folds.

        :param fold_i: the number of the current fold
        :param n_folds: the total number of folds
        :return: the data for the specified fold
        """
        raise NotImplementedError()


class Model(ComponentBase):
    def __init__(self, config, config_global, logger):
        super(Model, self).__init__(config, config_global, logger)
        self.__summary = None

        self.special_print = OrderedDict()
        # a dictionary to store special tensors that should be printed after each epoch

    def build(self, data, sess):
        raise NotImplementedError()


class Training(ComponentBase):
    def start(self, model, data, sess, evaluation):
        """

        :param model:
        :type model: Model
        :param data:
        :type data: Data
        :type evaluation: Evaluation
        """
        raise NotImplementedError()

    def restore_best_epoch(self, model, data, sess, evaluation):
        raise NotImplementedError()

    def remove_checkpoints(self):
        """Removes all the persisted checkpoint data that was generated during training for restoring purposes"""
        raise NotImplementedError()


class Evaluation(ComponentBase):
    def start(self, model, data, sess, valid_only=False):
        """

        :type model: Model
        :type data: Data
        :type sess: tensorflow.Session
        :type valid_only: bool
        :return: The score of the primary measure for all tested data splits
        :rtype: dict[basestring, float]
        """
        raise NotImplementedError()
