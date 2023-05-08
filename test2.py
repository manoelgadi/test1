import random

TRAIN_FILE = "synthetic_data_train.txt"
TEST_FILE = "synthetic_data_test.txt"
LEARNING_RATE = 0.1

class Perceptron:
    def __init__(self, num_features):
        self.weights = [random.uniform(-1, 1) for i in range(num_features)]
        self.bias = random.uniform(-1, 1)
    
    def feed_forward(self, inputs):
        output = self.bias
        for i in range(len(inputs)):
            output += self.weights[i] * inputs[i]
        return 1 if output > 0 else 0
    
    def train(self, inputs, targets):
        num_errors = 0
        for i in range(len(inputs)):
            x = inputs[i]
            norm_x = [x[j] / abs(x[j]) if x[j] != 0 else 0 for j in range(len(x))]
            y_pred = self.feed_forward(norm_x)
            if y_pred != targets[i]:
                num_errors += 1
                for j in range(len(norm_x)):
                    self.weights[j] += LEARNING_RATE * targets[i] * norm_x[j]
                self.bias += LEARNING_RATE * targets[i]
        return num_errors
    
    def evaluate(self, inputs, targets):
        confusion_matrix = [[0, 0], [0, 0]]
        for i in range(len(inputs)):
            x = inputs[i]
            y_pred = self.feed_forward(x)
            confusion_matrix[targets[i]][y_pred] += 1
        return confusion_matrix

def read_data(file_name):
    inputs = []
    targets = []
    with open(file_name, "r") as f:
        for line in f:
            data = line.strip().split(",")
            inputs.append([float(data[0]), float(data[1]), float(data[2])])
            targets.append(int(data[3]))
    return inputs, targets

confusion_avg = [[0, 0], [0, 0]]
num_runs = 10
for run in range(num_runs):
    perceptron = Perceptron(3)
    train_inputs, train_targets = read_data(TRAIN_FILE)
    num_epochs = 100
    total_errors = 0
    for i in range(num_epochs):
        num_errors = perceptron.train(train_inputs, train_targets)
        total_errors += num_errors
    test_inputs, test_targets = read_data(TEST_FILE)
    confusion_matrix = perceptron.evaluate(test_inputs, test_targets)
    for i in range(2):
        for j in range(2):
            confusion_avg[i][j] += confusion_matrix[i][j]

for i in range(2):
    for j in range(2):
        confusion_avg[i][j] /= num_runs
print(confusion_avg)
