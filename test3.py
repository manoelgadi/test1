import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_test_split(df, test_size=0.2):
    mask = np.random.rand(len(df)) < test_size
    train_df = df[~mask]
    test_df = df[mask]
    return train_df, test_df


def perceptron(train_X, train_y, test_X, test_y, learning_rate=0.1, max_iter=100):
    weights = np.random.rand(train_X.shape[1] + 1)
    train_X_bias = np.c_[np.ones((train_X.shape[0], 1)), train_X]
    test_X_bias = np.c_[np.ones((test_X.shape[0], 1)), test_X]
    misclassified = []
    for i in range(max_iter):
        shuffle_idx = np.random.permutation(len(train_X_bias))
        train_X_bias = train_X_bias[shuffle_idx]
        train_y = train_y[shuffle_idx]
        for j in range(len(train_X_bias)):
            z = np.dot(weights, train_X_bias[j])
            y_pred = 1 if z > 0 else 0
            if y_pred != train_y[j]:
                weights += learning_rate * (train_y[j] - y_pred) * train_X_bias[j]
        train_y_pred = np.where(np.dot(train_X_bias, weights) > 0, 1, 0)
        train_acc = np.sum(train_y_pred == train_y) / len(train_y)
        misclassified_train = np.sum(train_y_pred != train_y)
        test_y_pred = np.where(np.dot(test_X_bias, weights) > 0, 1, 0)
        test_acc = np.sum(test_y_pred == test_y) / len(test_y)
        misclassified_test = np.sum(test_y_pred != test_y)
        print(f"Iteration {i}: Train Acc = {train_acc:.3f}, Test Acc = {test_acc:.3f}")
        misclassified.append(misclassified_train)
    plt.plot(misclassified[:100])
    plt.xlabel('Iteration')
    plt.ylabel('Number of Misclassified Examples')
    plt.show()
    return weights


def calc_confusion_matrix(y_true, y_pred):
    true_pos = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    false_neg = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    false_pos = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    true_neg = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    return true_pos, false_neg, false_pos, true_neg


if __name__ == '__main__':
    train_data = pd.read_csv("datatraining.txt")
    test_data = pd.read_csv("datatset2.txt")
    train_df, test_df = train_test_split(train_data)

    features = ['Temperature', 'Light', 'CO2']
    label = 'Occupancy'

    train_X = train_df[features].to_numpy()
    train_y = train_df[label].to_numpy()
    test_X = test_df[features].to_numpy()
    test_y = test_df[label].to_numpy()

    weights = perceptron(train_X, train_y, test_X, test_y)

    true_pos, false_neg, false_pos, true_neg = calc_confusion_matrix(test_y, np.where(np.dot(np.c_[np.ones((test_X.shape[0], 1)), test_X], weights) > 0, 1, 0))
    print(f"True Positive: {true_pos}, False Negative: {false_neg}, False Positive: {false_pos}, True Negative: {true_neg}")