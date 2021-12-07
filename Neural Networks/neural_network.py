import pandas as pd
import numpy as np


def read_bank_data():
    columns = ['var', 'skew', 'kurt', 'entropy', 'label']
    train = pd.read_csv('./data/train.csv', header=None)
    train.columns = columns
    train.label.replace(0, -1, inplace=True)
    test = pd.read_csv('./data/test.csv', header=None)
    test.columns = columns
    test.label.replace(0, -1, inplace=True)
    train.insert(0, 'bias', 1)
    test.insert(0, 'bias', 1)

    return train.drop(columns=['label']).values, train.label.values, test.drop(columns=['label']).values, test.label.values


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNet:
    def __init__(self, layer_dimensions, num_epochs=100, lr=.01, d=.1, gamma=.1, zero_weights = False):
        self.input_size = layer_dimensions[0]
        self.output_size = layer_dimensions[-1]

        self.lr = lr
        self.d = d

        self.num_epochs = num_epochs
        self.gamma = gamma

        self.layer_dimensions = layer_dimensions
        self.layers = len(layer_dimensions)

        self.weights = [None]
        self.grads = [None]
        self.output = []

        for i in range(0, self.layers):
            if i > 0:
                input_dim = self.layer_dimensions[i] - 1
                output_dim = self.layer_dimensions[i - 1]
                if i == self.layers - 1:
                    input_dim += 1

                if not zero_weights:
                    weight = np.random.normal(size=(input_dim, output_dim))
                else:
                    weight = np.zeros((input_dim, output_dim))

                self.weights.append(weight)
                self.grads.append(np.zeros([input_dim, output_dim]))

            self.output.append(np.ones((self.layer_dimensions[i], 1)))

    def train(self, train_X, train_y):
        for epoch in range(self.num_epochs):
            random_idx = np.random.permutation(train_X.shape[0])
            for x, y in zip(train_X[random_idx], train_y[random_idx]):
                self(x.reshape((self.input_size, 1)))
                self.backwards(y.reshape((self.output_size, 1)))
                self.update_weights(self.gamma / (1 + ((self.gamma / self.d) * epoch)))

    def update_weights(self, lr):
        self.weights = [None] + [weight - (grad * lr) for grad, weight in zip(self.grads[1:], self.weights[1:])]

    def __call__(self, x):
        self.output[0] = x
        for i in range(1, self.layers):
            layer_output = self.weights[i] @ self.output[i - 1]
            if i < self.layers - 1:
                self.output[i][:-1, :] = sigmoid(layer_output).reshape((-1, 1))
            else:
                self.output[i] = layer_output

        return self.output[-1]

    def backwards(self, y):
        dL_dZ = self.output[-1] - y
        dZ_dW = np.tile(self.output[-2], (1, self.layer_dimensions[-1])).T

        self.grads[-1] = dL_dZ * dZ_dW
        dZ_dZ = self.weights[-1][:, :-1]

        for i in range(1, self.layers - 1)[::-1]:
            dL_dZ, dZ_dZ = self.layer_backwards(dL_dZ, dZ_dZ, i)

    def layer_backwards(self, dL_dZ, dZ_dZ, layer_index):
        no_bias_dim = self.layer_dimensions[layer_index] - 1

        layer_input = self.output[layer_index - 1]
        layer_output = self.output[layer_index][:-1]

        dL_dZ = dZ_dZ.T @ dL_dZ
        dZ_dZ = (layer_output * (1 - layer_output) * self.weights[layer_index])[:, :-1]

        self.grads[layer_index] = dL_dZ * layer_output * (1 - layer_output) * np.tile(layer_input, (1, no_bias_dim)).T
        return dL_dZ, dZ_dZ

    def predict(self, X):
        return np.concatenate([self.__call__(x.reshape(self.input_size)).T for x in X], axis=0)


if __name__=='__main__':
    print("Neural Network random weight init results:\n")
    train_X, train_y, test_x, test_y = read_bank_data()
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    input_size = train_X.shape[1]
    output_size = 1

    widths = [5, 10, 25, 50, 100]
    output = []
    for width in widths:
        model = NeuralNet([input_size, width, width, output_size])

        model.train(train_X.reshape((-1, input_size)), train_y)

        predictions = model.predict(train_X)
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = -1

        train_error = 1 - np.mean(predictions == train_y)

        predictions = model.predict(test_x)
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = -1

        test_error = 1 - np.mean(predictions == test_y)
        output.append([width, train_error.round(3), test_error.round(3)])
        print(f"width: {width}, train error: {train_error.round(3)}, test error: {test_error.round(3)}")

    output_df = pd.DataFrame(output, columns=['Width', 'Train Error', 'Test Error'])
    output_df.to_latex('latex_output_nonzero.txt', index=False)

    print("\n\nNeural Network zero weight init results:\n")
    output = []
    for width in widths:
        model = NeuralNet([input_size, width, width, output_size], zero_weights=True)

        model.train(train_X.reshape((-1, input_size)), train_y)

        predictions = model.predict(train_X)
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = -1

        train_error = 1 - np.mean(predictions == train_y)

        predictions = model.predict(test_x)
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = -1

        test_error = 1 - np.mean(predictions == test_y)
        output.append([width, train_error.round(3), test_error.round(3)])
        print(f"width: {width}, train error: {train_error.round(3)}, test error: {test_error.round(3)}")

    output_df = pd.DataFrame(output, columns=['Width', 'Train Error', 'Test Error'])
    output_df.to_latex('latex_output_zero.txt', index=False)
