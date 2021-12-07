from numpy.lib.arraysetops import isin
import torch
import pandas as pd
import numpy as np
import torch.nn as nn

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

class NeuralNet(nn.Module):
    
    def __init__(self, layer_dimensions, num_epochs=1000, ReLU=True):
        super(NeuralNet, self).__init__()      
        self.num_epochs = num_epochs          

        self.weights_init = nn.init.xavier_normal_ if not ReLU else nn.init.kaiming_uniform_
        def init_weights(model):
            if isinstance(model, nn.Linear):
                self.weights_init(model.weight)
                torch.nn.init.ones_(model.bias)
                        
        self.model = nn.Sequential(                       
            nn.Linear(layer_dimensions[0], layer_dimensions[1]),
            nn.Tanh() if not ReLU else nn.ReLU(),
            nn.Linear(layer_dimensions[1], layer_dimensions[2]),
            nn.Tanh() if not ReLU else nn.ReLU(),
            nn.Linear(layer_dimensions[2], layer_dimensions[3])        
        )

        self.model.apply(init_weights)

    def forward(self, X):    
        input_X = np.float32(X.copy())           
        out = torch.from_numpy(input_X)
        out.requires_grad = True        
        return self.model(out)
            

    def train(self, train_X, train_y):
        train_y = np.float32(train_y.copy())
        train_y = torch.from_numpy(train_y)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters())
        
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            output = self.forward(train_X)
            loss = criterion(output, train_y)
            loss.backward()
            optimizer.step()
    

if __name__=='__main__':
    train_X, train_y, test_x, test_y = read_bank_data()
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    input_size = train_X.shape[1]
    output_size = 1

    print("\nNeural Network pytorch ReLU results:\n")
    widths = [5, 10, 25, 50, 100]
    output = []
    for width in widths:
        model = NeuralNet([input_size, width, width, output_size], ReLU=True)
        
        model.train(train_X.reshape((-1, input_size)), train_y)
        predictions = model.forward(train_X).detach().numpy()
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = -1

        train_error = 1 - np.mean(predictions == train_y)

        predictions = model.forward(test_x).detach().numpy()
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = -1

        test_error = 1 - np.mean(predictions == test_y)
        output.append([width, train_error.round(3), test_error.round(3)])
        print(f"width: {width}, train error: {train_error.round(3)}, test error: {test_error.round(3)}")

    output_df = pd.DataFrame(output, columns=['Width', 'Train Error', 'Test Error'])
    output_df.to_latex('latex_pytorch_ReLU_output.txt', index=False)

    print("\nNeural Network pytorch tanh results:\n")
    output = []
    for width in widths:
        model = NeuralNet([input_size, width, width, output_size], ReLU=False)
        
        model.train(train_X.reshape((-1, input_size)), train_y)
        predictions = model.forward(train_X).detach().numpy()
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = -1

        train_error = 1 - np.mean(predictions == train_y)

        predictions = model.forward(test_x).detach().numpy()
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = -1

        test_error = 1 - np.mean(predictions == test_y)
        output.append([width, train_error.round(3), test_error.round(3)])
        print(f"width: {width}, train error: {train_error.round(3)}, test error: {test_error.round(3)}")

    output_df = pd.DataFrame(output, columns=['Width', 'Train Error', 'Test Error'])
    output_df.to_latex('latex_pytorch_tanh_output.txt', index=False)