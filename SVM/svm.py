import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def read_bank_data():
    columns = ['var', 'skew', 'kurt', 'entropy', 'label']
    train = pd.read_csv('./bank-note/train.csv', header=None)
    train.columns = columns
    train.label.replace(0, -1, inplace=True)
    test = pd.read_csv('./bank-note/test.csv', header=None)
    test.columns = columns
    test.label.replace(0, -1, inplace=True)
    train.insert(0, 'bias', 1)
    test.insert(0, 'bias', 1)

    return train.drop(columns=['label']), train.label, test.drop(columns=['label']), test.label


def svm_predict(weights, test_x):
    return [np.sign(weights.T @ x) for x in test_x.values]


def gaussian_kernel(x_1, x_2, gamma):
    return np.exp((-(np.linalg.norm(x_1-x_2)**2))/gamma)


def gauss_dual_svm_predict(train_x, test_x, alpha, train_y, gamma):
    train_x = train_x.values
    test_x = test_x.values
    train_y = train_y.values

    predictions = []
    for i in range(test_x.shape[0]):
        sum = 0
        for j in range(train_x.shape[0]):
            sum += alpha[j] * train_y[j] * gaussian_kernel(train_x[j], test_x[i], gamma)

        predictions.append(np.sign(sum))

    return predictions


def dual_svm_predict(weights, bias, test_x):
    return [np.sign(weights.T @ x + bias) for x in test_x.values]


def primal_svm_cost(weights, y, C, x):
    w_0 = weights.copy()
    w_0[0] = 0

    total_cost = .5 * (w_0.T @ w_0)
    for i in range(x.shape[0]):
        total_cost += np.max((0, 1 - y[i] * weights.T @ x[i]))

    return total_cost


def primal_stochastic_svm(train_x, train_y, test_x, test_y, C=100/873, lr_0=.001, a=.001):
    weights = np.zeros(train_x.shape[1])
    max_epochs = 100

    if a > 0:
        lr_schedule = lambda t,alpha: lr_0 / (1 + (lr_0/alpha) * t)
    else:
        lr_schedule = lambda t,alpha: lr_0 / (1 + t)

    convergence_series = []
    for epoch in range(1, max_epochs+1):
        lr = lr_schedule(epoch, a)
        rand_indices = np.random.choice(train_x.shape[0], train_x.shape[0],replace=True)
        batch_x = train_x.iloc[rand_indices]
        batch_y = train_y.iloc[rand_indices]
        for x, y in zip(batch_x.values, batch_y.values):
            w_0 = weights.copy()
            w_0[0] = 0
            if 1 - y * weights.T @ x <= 0:                
                weights -= lr * w_0
            else:
                weights -= lr * (w_0 - C * train_x.shape[0] * y * x)

        convergence_series.append(primal_svm_cost(weights, train_y.values, C, train_x.values))

    train_error = 1 - np.mean(svm_predict(weights, train_x) == train_y)
    test_error = 1 - np.mean(svm_predict(weights, test_x) == test_y)        

    plt.figure(figsize=(12,8))
    if a > 0:
        plt.title(f"SVM Cost vs Epoch for C: {C.round(3)} with Alpha: {a} lr_0: {lr_0}")
    else:
        plt.title(f"SVM Cost vs Epoch for C: {C.round(3)} lr_0: {lr_0}")
    plt.plot(np.arange(max_epochs), convergence_series)
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    if a > 0:
        plt.savefig(f"primal_svm_cost_C_{C.round(3)}_a_{a}.png")
    else:
        plt.savefig(f"primal_svm_cost_C_{C.round(3)}.png")

    return weights, train_error, test_error


def get_minimized_alphas(train_x, C, x, y, dual_svm_objective, dual_svm_jac):
    #a_0 = np.random.uniform(low=0, high=C, size=train_x.shape[0])
    a_0 = np.zeros(train_x.shape[0])
    constraints = ({'type': 'eq', 'fun': lambda x_: np.sum(x_*y), 'jac': lambda x_: y})
    bounds = np.array(x.shape[0] * [(0,C)])
    result = minimize(dual_svm_objective, a_0, constraints=constraints, method='SLSQP', bounds=bounds, jac=dual_svm_jac)

    minimized_a = result.x
    minimized_a[np.isclose(minimized_a, 0, atol=.001)] = 0
    minimized_a[np.isclose(minimized_a, C, atol=.001)] = C
    return minimized_a


def dual_svm(train_x, train_y, test_x, test_y, C=500/873, gamma=-1):
    x = train_x.values
    y = train_y.values
    x_prod = x @ x.T * (y * y[:, np.newaxis])
    
    if gamma > 0:
        x_prod = x * x[:, np.newaxis]
        x_prod = np.exp((-(np.sum(np.square(x_prod), axis=2)) / gamma)) * (y * y[:, np.newaxis])

    def dual_svm_objective(a):              
        return .5 * (a.T @ (x_prod @ a )) - np.sum(a)

    def dual_svm_jac(a):
        return (a.T @ x_prod) - np.ones(a.shape[0])

    minimized_a = get_minimized_alphas(train_x, C, x, y, dual_svm_objective, dual_svm_jac)

    weights = (minimized_a * x.T @ y)

    if gamma > 0:
        weights /= minimized_a.shape[0]

    bias = None
    bias_sum = 0
    for i in range(minimized_a.shape[0]):
        bias_sum += y[i] - weights.T @ x[i]

    bias = bias_sum / minimized_a.shape[0]

    if gamma > 0:
        train_preds = gauss_dual_svm_predict(train_x, train_x, minimized_a, train_y, gamma)
        test_preds = gauss_dual_svm_predict(train_x, test_x, minimized_a, train_y, gamma)
    else:
        train_preds = dual_svm_predict(weights, bias, train_x)
        test_preds = dual_svm_predict(weights, bias, test_x)


    test_error = 1 - np.mean(test_preds == test_y.values)
    train_error = 1 - np.mean(train_preds == train_y.values)

    return weights, bias, train_error, test_error, minimized_a



if __name__ == '__main__':
    train_x, train_y, test_x, test_y = read_bank_data()
    
    print("Primal SVM with stochastic sub-gradient descent with alpha learning-rate schedule:")
    results = []

    for C in np.array([100/873, 500/873, 700/873]):
        weights, train_error, test_error = primal_stochastic_svm(train_x, train_y, test_x, test_y, C, a=.0001)
        print(f"C: {C.round(3)}, Weights: {weights.round(3)}, train_error: {train_error.round(3)}, test_error: {test_error.round(3)}")
        results.append([C.round(3), train_error.round(3), test_error.round(3), weights.round(3)])

    df = pd.DataFrame(results, columns=['C', 'Train Error', 'Test Error', 'Weights'])
    df.to_latex('alpha_learning_rate_latex', index=False)

    print("\nPrimal SVM with stochastic sub-gradient descent without alpha learning-rate schedule:")
    results = []

    for C in np.array([100/873, 500/873, 700/873]):
        weights, train_error, test_error = primal_stochastic_svm(train_x, train_y, test_x, test_y, C, a=-1)
        print(f"C: {C.round(3)}, Weights: {weights.round(3)}, train_error: {train_error.round(3)}, test_error: {test_error.round(3)}")
        results.append([C.round(3), train_error.round(3), test_error.round(3), weights.round(3)])

    df = pd.DataFrame(results, columns=['C', 'Train Error', 'Test Error', 'Weights'])
    df.to_latex('no_alpha_learning_rate_latex', index=False)

    results = []
    print("\nDual SVM Linear Kernel")    
    for C in np.array([100/873, 500/873, 700/873]):
        weights, bias, train_error, test_error, _ = dual_svm(train_x.drop(columns=['bias']), train_y, test_x.drop(columns=['bias']), test_y, C)
        print(f"C: {C.round(3)}, bias: {bias.round(3)}, Weights: {weights.round(3)}, train_error: {train_error.round(3)}, test_error: {test_error.round(3)}")
        results.append([C.round(3), train_error.round(3), test_error.round(3), weights.round(3), bias.round(3)])

    df = pd.DataFrame(results, columns=['C', 'Train Error', 'Test Error', 'Weights', 'bias'])
    df.to_latex('dual_linear_kernel_latex', index=False)

    results = []
    support_results = []      
    print("\nDual SVM Gaussian Kernel")    
    for C in np.array([100/873, 500/873, 700/873]):
        prev_minimized_alphas = None
        num_overlap_support = None
        for gamma in np.array([.1, .5, 1, 5, 100]):
            weights, bias, train_error, test_error, minimized_alphas = dual_svm(train_x.drop(columns=['bias']), train_y, test_x.drop(columns=['bias']), test_y, C, gamma)
            num_support_vector = np.count_nonzero(minimized_alphas)
            print(f"C: {C.round(3)}, gamma: {gamma}, bias: {bias.round(3)}, Weights: {weights.round(3)}, train_error: {train_error.round(3)}, test_error: {test_error.round(3)}, Support Vector Count: {num_support_vector}")
            results.append([C.round(3), gamma, train_error.round(3), test_error.round(3)])
            if prev_minimized_alphas is not None:
                support_vectors = np.nonzero(minimized_alphas)[0]
                prev_support_vectors = np.nonzero(prev_minimized_alphas)[0]
                num_overlap_support = np.intersect1d(support_vectors, prev_support_vectors, assume_unique=True).shape[0]
                print(f"C: {C.round(3)}, gamma: {gamma}, Overlapping Support Vector Count: {num_overlap_support}")
            
            support_results.append([C.round(3), gamma, num_support_vector, num_overlap_support])
            prev_minimized_alphas = minimized_alphas

    df = pd.DataFrame(results, columns=['C', 'Gamma', 'Train Error', 'Test Error'])
    df.to_latex('dual_gaussian_kernel_latex', index=False)
    df = pd.DataFrame(support_results, columns=['C', 'Gamma', 'Number of Support Vectors', 'Number of Overlapping Support Vectors'])
    df.to_latex('dual_gaussian_support_vectors_latex', index=False)
