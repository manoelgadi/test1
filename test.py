import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            x, y, z, c = line.strip().split(',')
            data.append([float(x), float(y), float(z), int(c)])
    return data

def count_misclassified(w, b, data):
    """
    Count the number of misclassified data points in the given dataset
    using the weight vector w and bias term b.

    Args:
        w (list[float]): The weight vector.
        b (float): The bias term.
        data (list[list[float, int]]): The dataset, where each data point is
            represented as a list of [x, y, z, class].

    Returns:
        int: The number of misclassified data points.
    """
    misclassified = 0
    for x, y, z, c in data:
        if c == 1 and np.dot(w, [x, y, z]) + b <= 0:
            misclassified += 1
        elif c == 0 and np.dot(w, [x, y, z]) + b > 0:
            misclassified += 1
    return misclassified

def perceptron(data):
    """
    Train a perceptron model on the given dataset.

    Args:
        data (list[list[float, int]]): The dataset, where each data point is
            represented as a list of [x, y, z, class].

    Returns:
        tuple: A tuple of the weight vector and bias term for the trained model.
    """
    # Initialize the weight vector and bias term
    w = np.random.rand(3)
    b = np.random.rand()

    # Iterate until all data points are correctly classified
    iterations = 0
    misclassified_history = []
    while count_misclassified(w, b, data) > 0 and iterations < 100:
        for x, y, z, c in data:
            if c == 1 and np.dot(w, [x, y, z]) + b <= 0:
                w += np.array([x, y, z])/np.linalg.norm([x, y, z])
                b += 1
            elif c == 0 and np.dot(w, [x, y, z]) + b > 0:
                w -= np.array([x, y, z])/np.linalg.norm([x, y, z])
                b -= 1
        iterations += 1

        # Record the number of misclassified data points every iteration
        misclassified = count_misclassified(w, b, data)
        misclassified_history.append(misclassified)
        print(f"Iteration {iterations}: {misclassified} misclassified")

        # Stop iterating if the maximum number of iterations is reached
        if iterations >= 100:
            break

    # Plot the misclassified data points over iterations
    plt.plot(range(1, iterations+1), misclassified_history)
    plt.title("Misclassified Data Points over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Number of Misclassified Data Points")
    plt.show()

    return w, b

# Load the training data and train the perceptron model
train_data = load_data("synthetic_data_train.txt")
w, b = perceptron(train_data)

# Load the test data and evaluate the model
test_data = load_data("synthetic_data_test.txt")

# Count theCount the number of misclassified data points in the test dataset
test_misclassified = count_misclassified(w, b, test_data)
print(f"Number of misclassified test data points: {test_misclassified}")
