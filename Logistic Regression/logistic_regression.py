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


def logistic_regression_MAP(v):
    max_epochs = 100
    lr_0 = .001
    a = .001

    lr_schedule = lambda t,alpha: lr_0 / (1 + (lr_0/alpha) * t)

    train_x, train_y, test_x, test_y = read_bank_data()

    weights = np.zeros(train_x.shape[1])

    for epoch in range(1, max_epochs+1):
        lr = lr_schedule(epoch, a)
        rand_indices = np.random.choice(train_x.shape[0], train_x.shape[0],replace=True)
        batch_x = train_x[rand_indices]
        batch_y = train_y[rand_indices]
        for x, y in zip(batch_x, batch_y):
            weights -= lr * (-x.shape[0] * y * x / (1 + np.exp(y * np.sum(weights @ x))) + weights / v)

    y_preds = log_regression_predict(test_x, weights)
    test_error = 1 - np.mean(y_preds == test_y)
    y_preds = log_regression_predict(train_x, weights)
    train_error = 1 - np.mean(y_preds == train_y)

    return train_error, test_error

def logistic_regression_ML():
    max_epochs = 100
    lr_0 = .001
    a = .001

    lr_schedule = lambda t,alpha: lr_0 / (1 + (lr_0/alpha) * t)

    train_x, train_y, test_x, test_y = read_bank_data()

    weights = np.zeros(train_x.shape[1])

    for epoch in range(1, max_epochs+1):
        lr = lr_schedule(epoch, a)
        rand_indices = np.random.choice(train_x.shape[0], train_x.shape[0],replace=True)
        batch_x = train_x[rand_indices]
        batch_y = train_y[rand_indices]
        for x, y in zip(batch_x, batch_y):
            weights -= lr * (-x.shape[0] * y * x / (1 + np.exp(y * np.sum(weights @ x))))

    y_preds = log_regression_predict(test_x, weights)
    test_error = 1 - np.mean(y_preds == test_y)
    y_preds = log_regression_predict(train_x, weights)
    train_error = 1 - np.mean(y_preds == train_y)

    return train_error, test_error

def log_regression_predict(test_x, weights):
    y_preds = []
    for x in test_x:
        y_preds.append(weights.T @ x)

    y_preds = np.array(y_preds)

    y_preds[y_preds >= 0] = 1
    y_preds[y_preds < 0] = -1
    return y_preds
    

if __name__ == "__main__":
    print("Logistic Regression MAP results:\n")
    variances = [0.01,0.1,0.5,1,3,5,10,100]
    output = []
    for v in variances:
        train_error, test_error = logistic_regression_MAP(v)
        print(f"v: {v}, train error: {train_error.round(3)}, test error: {test_error.round(3)}")
        output.append([v, train_error, test_error])

    output_df = pd.DataFrame(output, columns=['v', 'Train Error', 'Test Error'])
    output_df.to_latex('logistic_regression_map.txt', index=False)

    print("\nLogistic Regression ML results:\n")
    train_error, test_error = logistic_regression_ML()
    print(f"ML: train error: {train_error.round(3)}, test error: {test_error.round(3)}")