from experiment.qa.data import QAData
from experiment.qa.data.insuranceqa.reader.json_reader import JSONArchiveReader


class JSONData(QAData):
    def _get_train_readers(self):
        return [JSONArchiveReader(self.config['insuranceqa'], self.lowercased, self.logger)]


component = JSONData
