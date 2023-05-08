import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            x, y, z, c = line.strip().split(',')
            data.append([float(x), float(y), float(z), int(c)])
    return data

def train_perceptron(data):
    w = np.random.rand(3)
    b = np.random.rand()
    iterations = 0
    mistakecount_history = []
    while mistake_count(w, b, data) > 0 and iterations < 100:
        for x, y, z, c in data:
            if c == 1 and np.dot(w, [x, y, z]) + b <= 0:
                w += np.array([x, y, z])/np.linalg.norm([x, y, z])
                b += 1
            elif c == 0 and np.dot(w, [x, y, z]) + b > 0:
                w -= np.array([x, y, z])/np.linalg.norm([x, y, z])
                b -= 1
        iterations += 1
        mistakecount = mistake_count(w, b, data)
        mistakecount_history.append(mistakecount)
        print(f"Iteration {iterations}: {mistakecount} misclassified")
        if iterations >= 100:
            break
    plt.plot(range(1, iterations+1), mistakecount_history)
    plt.xlabel("Iteration")
    plt.ylabel("Number of data that are not properly classified")
    plt.show()
    return w, b

def mistake_count(w, b, data):
    mistakecount = 0
    for x, y, z, c in data:
        if c == 1 and np.dot(w, [x, y, z]) + b <= 0:
            mistakecount += 1
        elif c == 0 and np.dot(w, [x, y, z]) + b > 0:
            mistakecount += 1
    return mistakecount

def evaluate_perceptron(w, b, data):
    misclassified = mistake_count(w, b, data)
    print(f"Number of misclassified data points: {misclassified}")

train_data = load_data("synthetic_data_train.txt")
w, b = train_perceptron(train_data)

test_data = load_data("synthetic_data_test.txt")
evaluate_perceptron(w, b, test_data)
