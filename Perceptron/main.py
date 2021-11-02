import pandas as pd
import numpy as np



def read_bank_data():
    columns = ['var', 'skew', 'kurt', 'entropy', 'label']
    train = pd.read_csv('./bank-note/train.csv', header=None)
    train.columns = columns
    train.label.replace(0, -1, inplace=True)
    test = pd.read_csv('./bank-note/test.csv', header=None)
    test.columns = columns
    test.label.replace(0 , -1, inplace=True)
    train.insert(0, 'bias', 1)
    test.insert(0, 'bias', 1)

    return train.drop(columns=['label']), train.label, test.drop(columns=['label']), test.label

def perceptron_predict(weights, test_x):
    predictions = np.zeros(test_x.shape[0])
    for i, x in enumerate(test_x.values):
        predictions[i] = np.sign(weights.T @ x).astype(int)

    return predictions

def voted_perceptron_predict(weights, votes, test_x):
    predictions = np.zeros(test_x.shape[0])
    for i, x in enumerate(test_x.values):
        predictions[i] = np.sign(np.sum([c * np.sign(w.T @ x) for w, c in zip(weights, votes)]))

    return predictions

def averaged_preceptron_predict(a, test_x):
    predictions = np.zeros(test_x.shape[0])
    for i,x in enumerate(test_x.values):
        predictions[i] = np.sign(a.T @ x)

    return predictions


def bank_perceptron(lr=.25, EPOCHS=10):
    train_x, train_y, test_x, test_y = read_bank_data()
    return perceptron(train_x, train_y, test_x, test_y, lr, EPOCHS)


def bank_voted_perceptron(lr=.25, EPOCHS=10):
    train_x, train_y, test_x, test_y = read_bank_data()
    return voted_perceptron(train_x, train_y, test_x, test_y, lr, EPOCHS)


def bank_averaged_preceptron(lr=.25, EPOCHS=10):
    train_x, train_y, test_x, test_y = read_bank_data()
    return averaged_perceptron(train_x, train_y, test_x, test_y, lr, EPOCHS)


def perceptron(train_x, train_y, test_x, test_y, lr, EPOCHS):
    weights = np.zeros(train_x.shape[1])
    for _ in range(EPOCHS):        
        random_indices = np.random.choice(train_x.shape[0], train_x.shape[0], replace=False)
        shuff_x = train_x.values[random_indices]
        shuff_y = train_y.values[random_indices]
        
        for x,y in zip(shuff_x, shuff_y):
            y_hat = np.sign(weights.T @ x).astype(int)
            if y_hat != y:            
                weights += lr * (y * x)

    accuracy = (np.mean(perceptron_predict(weights, test_x) == test_y))
    return weights, accuracy



def voted_perceptron(train_x, train_y, test_x, test_y, lr, EPOCHS):
    weights = np.zeros(train_x.shape[1])
    all_weights = []
    counts = []
    for _ in range(EPOCHS):
        random_indices = np.random.choice(train_x.shape[0], train_x.shape[0], replace=False)
        shuff_x = train_x.values[random_indices]
        shuff_y = train_y.values[random_indices]
        for x, y in zip(shuff_x, shuff_y):
            if y * weights.T @ x <= 0:
                old_weights = weights.copy()
                all_weights.append(old_weights)
                weights = old_weights + (lr * y * x)
                counts.append(1)            
            else:
                if len(counts) == 0:
                    counts.append(0)

                counts[-1] += 1

    all_weights = np.array(all_weights)
    counts = np.array(counts)
    predictions = voted_perceptron_predict(all_weights, counts, test_x)    
    accuracy = (np.mean(predictions == test_y))
    return list(zip(all_weights,counts)), accuracy



def averaged_perceptron(train_x, train_y, test_x, test_y, lr, EPOCHS):
    weights = np.zeros(train_x.shape[1])
    a = weights.copy()
    for _ in range(EPOCHS):
        random_indices = np.random.choice(train_x.shape[0], train_x.shape[0], replace=False)
        shuff_x = train_x.values[random_indices]
        shuff_y = train_y.values[random_indices]
        for x, y in zip(shuff_x, shuff_y):
            if y * weights.T @ x <= 0:
                weights += lr * y * x
            a += weights

    accuracy = (np.mean(averaged_preceptron_predict(a, test_x) == test_y))
    return weights, a, accuracy



if __name__ == '__main__':
    import sys
    if len(sys.argv) <= 1:
        weights, accuracy = bank_perceptron()
        print(f'Standard:')
        print(f'weights: {weights}')
        print(f'accuracy: {accuracy}')
        print()

        weights_and_votes, accuracy = bank_voted_perceptron()
        print(f'Voted:')
        for w, v in weights_and_votes:
            print(f'weight: {w}, vote: {v}')

        print(f'Accuracy: {accuracy}')
        print()

        final_weights, a_vector, accuracy = bank_averaged_preceptron()
        print('Averaged:')
        print(f'final weights: {final_weights}')
        print(f'a vector: {a_vector}')
        print(f'Accuracy: {accuracy}')
    else:
        if sys.argv[1] == 'voted':
            weights_and_votes, accuracy = bank_voted_perceptron()
            print(f'Voted:')
            for w, v in weights_and_votes:
                print(f'weight: {w}, vote: {v}')

            print(f'Accuracy: {accuracy}')
            print()

            # df = pd.DataFrame(np.vstack([w for w, _ in weights_and_votes]))
            # df.columns = ['bias', 'w1', 'w2', 'w3', 'w4']
            # df['vote'] = [v for _,v in weights_and_votes]
            # for row in df.values:
            #     print(' & '.join((str(round(x, 2)) for x in row.tolist())) + ' \\\\')


        elif sys.argv[1] == 'averaged':
            final_weights, a_vector, accuracy = bank_averaged_preceptron()
            print('Averaged:')
            print(f'final weights: {final_weights}')
            print(f'a vector: {a_vector}')
            print(f'Accuracy: {accuracy}')
        elif sys.argv[1] == 'standard':
            weights, accuracy = bank_perceptron()
            print(f'Standard:')
            print(f'weights: {weights}')
            print(f'accuracy: {accuracy}')
            print()
        else:
            weights, accuracy = bank_perceptron()
            print(f'Standard:')
            print(f'weights: {weights}')
            print(f'accuracy: {accuracy}')
            print()
