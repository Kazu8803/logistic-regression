from algorithm.regression.BinaryLogisticRegression import BinaryLogisticRegression
from utils.CsvFileHandler import CsvFileHandler
from utils.DataNormalization import DataNormalization
from utils.Matrix import Matrix

dn = DataNormalization()
csv_handler = CsvFileHandler()
blr = BinaryLogisticRegression()
m = Matrix()

# Initialize
data = csv_handler.load_dataset('{}{}'.format(
    "/home/dassuncao/Projects/logistic-regression/dataset/",
    "diabetes.csv")
)

# Normalize
data = dn.zscore(data)

# Start process
x = m.get_x(data)
y = m.get_y(data)
w = m.get_w(data)
result = blr.start_regression(x, w, y)

print("Result", result)
