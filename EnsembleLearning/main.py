from ID3 import ID3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_bank_data():
    label_values = ['yes', 'no']

    attr_values = {
        'age': [],
        'job': ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
                "blue-collar", "self-employed", "retired", "technician", "services"],
        'marital': ["married", "divorced", "single"],
        'education': ["unknown", "secondary", "primary", "tertiary"],
        'default': ['yes', 'no'],
        'balance': [],
        'housing': ['yes', 'no'],
        'loan': ['yes', 'no'],
        'contact': ['unknown', 'telephone', 'cellular'],
        'day': [],
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'duration': [],
        'campaign': [],
        'pdays': [],
        'previous': [],
        'poutcome': ['unknown', 'other', 'failure', 'success']
    }

    columns = list(attr_values.keys()) + ['label']
    train = pd.read_csv('./data/bank/train.csv', header=None)
    train.columns = columns
    train.label = train.label.map({'yes': 1, 'no': -1})
    test = pd.read_csv('./data/bank/test.csv', header=None)
    test.columns = columns
    test.label = test.label.map({'yes': 1, 'no': -1})
    return train, test, attr_values, label_values

def create_adaboost_table():
    train, test, attr_values, label_values = load_bank_data()

    weights = np.full(shape=train.shape[0], fill_value=1 / train.shape[0])

    num_stumps = 500
    test_predictions = np.zeros(shape=(num_stumps, test.shape[0]))
    train_predictions = np.zeros(shape=(num_stumps, train.shape[0]))

    overall_table ={'num_stumps': [], 'train_accuracy': [], 'test_accuracy': []}
    single_table ={'num_stumps': [], 'train_accuracy': [], 'test_accuracy': []}
    for i in range(num_stumps):
        stump = ID3(train, attr_values, label_values, 'IG', 2, weights, stump=True)
        predictions = np.array(stump.Predict(train))

        weighted_error = np.sum(weights[predictions != train.label])
        weighted_error = np.clip(weighted_error, 0+1e-20, 1-1e-20)
        a_t = .5 * np.log((1 - weighted_error) / weighted_error)

        weights[predictions != train.label] *= np.exp(a_t)
        weights[predictions == train.label] *= np.exp(-a_t)
        weights /= np.sum(weights)

        test_pred = stump.Predict(test)
        train_predictions[i, :] = np.array(predictions) * a_t
        test_predictions[i, :] = np.array(test_pred) * a_t

        final_train_preds = np.sign(np.sum(train_predictions, axis=0)).astype(int)
        final_test_preds = np.sign(np.sum(test_predictions, axis=0)).astype(int)

        overall_table['num_stumps'].append(i)
        overall_table['train_accuracy'].append(np.mean(final_train_preds == train.label))
        overall_table['test_accuracy'].append(np.mean(final_test_preds == test.label))

        single_table['num_stumps'].append(i)
        single_table['train_accuracy'].append(np.mean(predictions == train.label))
        single_table['test_accuracy'].append(np.mean(test_pred == test.label))

    table_df = pd.DataFrame.from_dict(overall_table)
    single_table_df = pd.DataFrame.from_dict(single_table)

    plt.plot(table_df.num_stumps, table_df.train_accuracy, label='Train Accuracy')
    plt.plot(table_df.num_stumps, table_df.test_accuracy, label='Test Accuracy')
    plt.title("Adaboost Final Hypothesis Accuracies")
    plt.xlabel('Number of Stumps')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(single_table_df.num_stumps, single_table_df.train_accuracy, label='Train Accuracy')
    plt.plot(single_table_df.num_stumps, single_table_df.test_accuracy, label='Test Accuracy')
    plt.title("Adaboost Stump Accuracies")
    plt.xlabel('Number of Stumps')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()

        
            
def create_bagged_tree_table():
    train, test, attr_values, label_values = load_bank_data()

    num_trees = 500
    test_predictions = np.zeros(shape=(num_trees, test.shape[0]))
    train_predictions = np.zeros(shape=(num_trees, train.shape[0]))
    overall_table = {'num_trees': [], 'train_accuracy': [], 'test_accuracy': []}

    for i in range(num_trees):
        train_subset = train.sample(train.shape[0], replace=True)
        tree = ID3(train_subset, attr_values, label_values, 'IG', None)

        train_predictions[i, :] = np.array(tree.Predict(train))
        test_predictions[i, :] = np.array(tree.Predict(test))
        final_train_preds = np.sign(np.sum(train_predictions, axis=0) / (i+1)).astype(int)
        final_test_preds = np.sign(np.sum(test_predictions, axis=0) / (i+1)).astype(int)

        final_train_acc = np.mean(final_train_preds == train.label)
        final_test_acc = np.mean(final_test_preds == test.label)
        print(f"i: {i}, train_accuracy: {final_train_acc}")
        print(f"i: {i}, test_accuracy: {final_test_acc}")
        print()

        overall_table['num_trees'].append(i)
        overall_table['train_accuracy'].append(final_train_acc)
        overall_table['test_accuracy'].append(final_test_acc)        

    table_df = pd.DataFrame.from_dict(overall_table)
    plt.plot(table_df.num_trees, table_df.train_accuracy, label='Train Accuracy')
    plt.plot(table_df.num_trees, table_df.test_accuracy, label='Test Accuracy')
    plt.title("Bagged Trees Final Hypothesis Accuracies")
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()



def create_bagged_tree_bias_variance_decomp_table():
    train, test, attr_values, label_values = load_bank_data()

    num_trees = 500
    num_iter = 100   
    single_tree_preds = []
    final_hyp_preds = []

    for j in range(num_iter):        
        test_predictions = np.zeros(shape=(num_trees, test.shape[0]))
        train_predictions = np.zeros(shape=(num_trees, train.shape[0]))
        for i in range(num_trees):            
            train_subset = train.sample(1000, replace=False)
            tree = ID3(train_subset, attr_values, label_values, 'IG', None)

            preds = tree.Predict(test)
            if i == 0:
                single_tree_preds.append(preds)

            train_predictions[i, :] = np.array(tree.Predict(train))
            test_predictions[i, :] = np.array(preds)
            final_train_preds = np.sign(np.sum(train_predictions, axis=0) / (i+1)).astype(int)
            final_test_preds = np.sign(np.sum(test_predictions, axis=0) / (i+1)).astype(int)

            if i == num_trees - 1:
                final_hyp_preds.append(final_test_preds)

            final_train_acc = np.mean(final_train_preds == train.label)
            final_test_acc = np.mean(final_test_preds == test.label)
            print(f"j: {j}, i: {i}, train_accuracy: {final_train_acc}")
            print(f"j: {j}, i: {i}, test_accuracy: {final_test_acc}")
            print()

    single_tree_preds = np.array(single_tree_preds)
    final_hyp_preds = np.array(final_hyp_preds)

    single_tree_bias = np.mean(np.square(test.label - np.mean(single_tree_preds, axis=0)))
    single_tree_variance = np.var(single_tree_preds)    

    final_trees_bias = np.mean(np.square(test.label - np.mean(final_hyp_preds, axis=0)))
    final_trees_var = np.var(final_hyp_preds)

    return single_tree_bias, single_tree_variance, final_trees_bias, final_trees_var

    


def create_random_forest_table():
    train, test, attr_values, label_values = load_bank_data()

    num_trees = 500
    test_predictions = np.zeros(shape=(num_trees, test.shape[0]))
    train_predictions = np.zeros(shape=(num_trees, train.shape[0]))

    table = {'train_2_accuracy': [], 'train_4_accuracy': [], 'train_6_accuracy': [],
            'test_2_accuracy': [], 'test_4_accuracy': [], 'test_6_accuracy': []}
    
    all_test_preds = []
    for attr_sample in [2, 4, 6]:
        for i in range(num_trees):        
                train_subset = train.sample(train.shape[0], replace=True)
                tree = ID3(train_subset, attr_values, label_values, 'IG', None, attribute_sample=attr_sample)

                train_predictions[i, :] = np.array(tree.Predict(train))
                test_predictions[i, :] = np.array(tree.Predict(test))
                final_train_preds = np.sign(np.sum(train_predictions, axis=0) / (i+1)).astype(int)
                final_test_preds = np.sign(np.sum(test_predictions, axis=0) / (i+1)).astype(int)
                all_test_preds.append(tree.Predict(test))

            
                final_train_acc = np.mean(final_train_preds == train.label)
                final_test_acc = np.mean(final_test_preds == test.label)
                print(f"i: {i}, feature count: {attr_sample}, train_accuracy: {final_train_acc}")
                print(f"i: {i}, feature count: {attr_sample}, test_accuracy: {final_test_acc}")

                table[f"train_{attr_sample}_accuracy"].append(final_train_acc)
                table[f"test_{attr_sample}_accuracy"].append(final_test_acc)
                print()


    print()
    for attr_sample in [2, 4, 6]:
        plt.title('Random Forest Final Hypothesis Accuracies')
        plt.plot(range(num_trees), table[f"train_{attr_sample}_accuracy"], label=f'Features: {attr_sample}, Train Accuracy')
        plt.plot(range(num_trees), table[f"test_{attr_sample}_accuracy"], label=f'Features: {attr_sample}, Test Accuracy')
        plt.legend(loc='best')
        plt.show()

        
    return table


#create_adaboost_table()
#create_bagged_tree_table()
create_random_forest_table()
#create_bagged_tree_bias_variance_decomp_table()
