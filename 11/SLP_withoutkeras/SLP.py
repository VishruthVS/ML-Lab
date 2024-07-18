import numpy as np

def step_function(x):
    return np.where(x >= 0, 1, 0)

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[0], [0], [0], [1]])

X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([[0], [1], [1], [1]])

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        self.weights = np.zeros((input_size, 1))
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, y):
        for _ in range(self.epochs):
            for inputs, label in zip(X, y):
                inputs = inputs.reshape(-1, 1)
                linear_output = np.dot(inputs.T, self.weights) + self.bias
                prediction = step_function(linear_output)
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return step_function(linear_output)

perceptron_and = Perceptron(input_size=2)
perceptron_and.train(X_and, y_and)

perceptron_or = Perceptron(input_size=2)
perceptron_or.train(X_or, y_or)

print("AND Function Predictions:")
print(perceptron_and.predict(X_and))

print("\nOR Function Predictions:")
print(perceptron_or.predict(X_or))

and_test_input = np.array([[1, 1]])
print("\nAND Function Prediction for input [1, 1]:")
print(perceptron_and.predict(and_test_input))

or_test_input = np.array([[0, 1]])
print("\nOR Function Prediction for input [0, 1]:")
print(perceptron_or.predict(or_test_input))
