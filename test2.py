import random

# 학습용 데이터 파일
TRAIN_FILE = "synthetic_data_train.txt"

# 테스트용 데이터 파일
TEST_FILE = "synthetic_data_test.txt"

# Learning rate
LEARNING_RATE = 0.1

# Perceptron 클래스
class Perceptron:
    def __init__(self, num_features):
        # Weight vector 초기화
        self.weights = [random.uniform(-1, 1) for i in range(num_features)]
        self.bias = random.uniform(-1, 1)
    
    def predict(self, x):
        # Activation function
        y = self.bias
        for i in range(len(x)):
            y += self.weights[i] * x[i]
        return 1 if y > 0 else 0
    
    def train(self, X, y):
        num_errors = 0
        for i in range(len(X)):
            x = X[i]
            # Normalize feature vector
            norm_x = [x[j] / abs(x[j]) if x[j] != 0 else 0 for j in range(len(x))]
            # Prediction
            y_pred = self.predict(norm_x)
            # Update weight vector and bias
            if y_pred != y[i]:
                num_errors += 1
                for j in range(len(norm_x)):
                    self.weights[j] += LEARNING_RATE * y[i] * norm_x[j]
                self.bias += LEARNING_RATE * y[i]
        return num_errors
    
    def test(self, X, y):
        confusion_matrix = [[0, 0], [0, 0]]
        for i in range(len(X)):
            x = X[i]
            y_pred = self.predict(x)
            confusion_matrix[y[i]][y_pred] += 1
        return confusion_matrix

# 데이터 읽어오기
def read_data(filename):
    X = []
    y = []
    with open(filename, "r") as f:
        for line in f:
            data = line.strip().split(",")
            X.append([float(data[0]), float(data[1]), float(data[2])])
            y.append(int(data[3]))
    return X, y

# 10회 실행하여 confusion matrix 계산
confusion_matrix_avg = [[0, 0], [0, 0]]
num_runs = 10
for run in range(num_runs):
    # Perceptron 객체 생성
    perceptron = Perceptron(3)

    # Training set 읽어오기
    train_X, train_y = read_data(TRAIN_FILE)

    # Training
    num_epochs = 100
    total_errors = 0
    for i in range(num_epochs):
        num_errors = perceptron.train(train_X, train_y)
        total_errors += num_errors

    # Test set 읽어오기
    test_X, test_y = read_data(TEST_FILE)

    # Test
    confusion_matrix = perceptron.test(test_X, test_y)
    for i in range(2):
        for j in range(2):
            confusion_matrix_avg[i][j] += confusion_matrix[i][j]

# 평균값 계산
for i in range(2):
    for j in range(2):
        confusion_matrix_avg[i][j] /= num_runs

# 결과 출력
print("Confusion matrix (averaged over {} runs):".format(num_runs))
print(confusion_matrix_avg)

