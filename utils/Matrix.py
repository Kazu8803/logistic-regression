from numpy import dot
from numpy import exp
from numpy import random


class Matrix:

    @staticmethod
    def dot(a, b):
        return dot(a, b)

    @staticmethod
    def exp(x):
        return exp(x)

    @staticmethod
    def get_x(data):
        return data.iloc[:, :-1]

    @staticmethod
    def get_y(data):
        return data.iloc[:, -1]

    @staticmethod
    def get_w(x):
        return random.rand(x.shape[0], 1)
