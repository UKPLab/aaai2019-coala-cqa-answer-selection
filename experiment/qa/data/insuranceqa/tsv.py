from experiment.qa.data import QAData
from experiment.qa.data.insuranceqa.reader.tsv_reader import TSVReader


class TSVData(QAData):
    def _get_train_readers(self):
        return [TSVReader(p, self.lowercased, self.logger, self.config.get('generated_questions')) for p in self.config['train_data']]

    def _get_transfer_readers(self):
        return [TSVReader(p, self.lowercased, self.logger, self.config.get('generated_questions')) for p in self.config.get('transfer_test', [])]


component = TSVData
