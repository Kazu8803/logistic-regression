from scipy.stats import zscore

from utils.Matrix import Matrix


class DataNormalization:

    def __init__(self, matrix=Matrix()):
        self.__matrix = matrix

    def zscore(self, data):
        normalized = self.__matrix.get_x(data).apply(zscore)
        classes = self.__matrix.get_y(data)
        return normalized.join(classes)
