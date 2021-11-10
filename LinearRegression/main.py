import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def load_concrete_data():    
    columns = ['Cement', 'Slag', 'Fly_ash', 'Water', 'SP', 'Coarse_Aggr', 'Fine_Aggr', 'SLUMP']
    train = pd.read_csv('./data/concrete/train.csv', header=None)
    train.columns = columns
    test = pd.read_csv('./data/concrete/test.csv', header=None)
    test.columns = columns

    return train.drop(columns=['SLUMP']), train.SLUMP, test.drop(columns=['SLUMP']), test.SLUMP


def compute_analytic_linear_regression():
    train_x, train_y, _, _ = load_concrete_data()    
    train_x.insert(0, 'bias', np.ones(shape=train_x.shape[0]))

    return np.linalg.inv((train_x.values.T @ train_x.values)) @ (train_x.values.T @ train_y.values)

def compute_linear_regression_cost(x, y, weights):
    return .5 * np.sum((y.values - (weights @ x.values.T))**2)

def linear_regression_concrete_batch():
    train_x, train_y, test_x, test_y = load_concrete_data()

    train_x.insert(0, 'bias', np.ones(shape=train_x.shape[0]))
    test_x.insert(0, 'bias', np.ones(shape=test_x.shape[0]))
    
    weights = np.zeros(shape=train_x.shape[1])
    train_costs = []
    test_costs = []
    i = 0 
    lr = 0.005
    diff_norm = 100
    while diff_norm > 10e-6:
        train_costs.append(compute_linear_regression_cost(train_x, train_y, weights))    
        grads = np.zeros(shape=weights.shape)        
        for x, y in zip(train_x.values, train_y.values):            
            err = y - weights.T @ x
            grads -= err * x
        
        diff_norm = np.linalg.norm(grads * lr, 2)
        weights -= lr * grads
        i+=1

        test_costs.append(compute_linear_regression_cost(test_x, test_y, weights))


    
    fig, ax1 = plt.subplots()
    plt.title(f'Linear Regression Batch Costs, Learning Rate: {lr}')
    ax1.plot(range(i), train_costs, label='Train Costs', c='blue')
    plt.ylabel('Train Cost')
    ax2 = ax1.twinx()
    ax2.plot(range(i), test_costs, label='Test Costs', c='orange')
    plt.ylabel('Test Cost')
    fig.legend(loc='upper left')
    plt.savefig('batch.png')
    plt.show()    
    return weights
    

def linear_regression_concrete_stochastic():
    train_x, train_y, test_x, test_y = load_concrete_data()

    train_x.insert(0, 'bias', np.ones(shape=train_x.shape[0]))
    test_x.insert(0, 'bias', np.ones(shape=test_x.shape[0]))

    weights = np.zeros(shape=train_x.shape[1])
    num_iter = 100000
    x_plot = np.arange(num_iter)
    y_plot = np.zeros(shape=num_iter)

    train_costs = []
    test_costs = []

    lr = 0.0002
    for i in range(num_iter):
        train_costs.append(compute_linear_regression_cost(train_x, train_y, weights))  
        idx = np.random.randint(low=0, high=train_x.shape[0])
        x = train_x.values[idx]
        y = train_y.values[idx]
                
        y_plot[i] = compute_linear_regression_cost(train_x, train_y, weights)
        err = y - weights.T @ x
        grads = -err * x
        weights -= lr * grads

        test_costs.append(compute_linear_regression_cost(test_x, test_y, weights))


    fig, ax1 = plt.subplots()
    plt.title(f'Linear Regression Stochastic Costs, Learning Rate: {lr}')
    ax1.plot(range(num_iter), train_costs, label='Train Costs', c='blue')
    plt.ylabel('Train Cost')
    ax2 = ax1.twinx()
    ax2.plot(range(num_iter), test_costs, label='Test Costs', c='orange')
    plt.ylabel('Test Cost')
    fig.legend(loc='upper left')
    plt.savefig('stochastic.png')
    plt.show()    
    return weights



print(linear_regression_concrete_batch())
print(linear_regression_concrete_stochastic())

