import numpy as np
import pandas as pd


train_data = pd.read_csv("datatraining.txt")
test_data = pd.read_csv("datatset2.txt")


features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
label = 'Occupancy'

X_train = train_data[features].to_numpy()
y_train = train_data[label].to_numpy()
X_test = test_data[features].to_numpy()
y_test = test_data[label].to_numpy()


class BinaryClassifier:
    def __init__(self, learning_rate=0.1, max_iter=100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.w = None
    
    def fit(self, X, y):
        self.w = np.random.rand(X.shape[1] + 1)
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        for i in range(self.max_iter):
            shuffle_idx = np.random.permutation(len(X_bias))
            X_bias = X_bias[shuffle_idx]
            y = y[shuffle_idx]
            for j in range(len(X_bias)):
                z = np.dot(self.w, X_bias[j])
                y_pred = 1 if z > 0 else 0
                if y_pred != y[j]:
                    self.w += self.learning_rate * (y[j] - y_pred) * X_bias[j]
    
    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        z = np.dot(X_bias, self.w)
        y_pred = np.where(z > 0, 1, 0)
        return y_pred


binary_classifier = BinaryClassifier()
binary_classifier.fit(X_train, y_train)

y_pred = binary_classifier.predict(X_test)

tp = np.sum(np.logical_and(y_test == 1, y_pred == 1))
fn = np.sum(np.logical_and(y_test == 1, y_pred == 0))
fp = np.sum(np.logical_and(y_test == 0, y_pred == 1))
tn = np.sum(np.logical_and(y_test == 0, y_pred == 0))

print(f"{'Actual 0':<5s}{tn:>12d}{fp:>12d}")
print(f"{'Actual 1':<5s}{fn:>12d}{tp:>12d}")
