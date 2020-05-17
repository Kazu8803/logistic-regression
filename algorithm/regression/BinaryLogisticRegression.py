from utils.Matrix import Matrix


class BinaryLogisticRegression:

    def __init__(self, matrix=Matrix(), it_max=1000, learning_rate=1e-3):
        self.__matrix = matrix
        self.__it_max = it_max
        self.__learning_rate = learning_rate

    def __sigmoid_active_function(self, x):
        return 1 / (1 + self.__matrix.exp(-x))

    def __sum_weights(self, x, w):
        return self.__matrix.dot(x, w)

    def __predict(self, x, w):
        return self.__sigmoid_active_function(self.__sum_weights(x, w))

    def __update_weights(self, w, g):
        return w - g * self.__learning_rate

    @staticmethod
    def __update_gradient(g, p, y, x):
        return g + (p - ((1 + y) / 2)) * x

    @staticmethod
    def __calc_gradient_norm(g):
        return float(g ** 2) ** 0.5

    def start_regression(self, x, w, y):
        it, g = 0, 0
        while it < self.__it_max:
            print("Iterations", it, "gradient norm", self.__calc_gradient_norm(g))
            p = self.__predict(x, w)
            g = self.__update_gradient(g, p, y, x)
            w = self.__update_weights(w, g)
