from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class Metrics:

    @staticmethod
    def accuracy(actual_classes, predicted_classes):
        return accuracy_score(actual_classes, predicted_classes)

    @staticmethod
    def confusion_matrix(actual_classes, predicted_classes):
        return confusion_matrix(actual_classes, predicted_classes)

    @staticmethod
    def classification_report(actual_classes, predicted_classes):
        return classification_report(actual_classes, predicted_classes)
