from pandas import read_csv
from sklearn.utils import shuffle


class CsvFileHandler:

    @staticmethod
    def load_dataset(path, header=None):
        dataset = read_csv(path, header=header)
        return shuffle(dataset)
