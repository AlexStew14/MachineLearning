from ID3 import ID3
import numpy as np
import pandas as pd





def create_car_table():
    label_values = ['unacc', 'acc', 'good', 'vgood']
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

    attr_values = {'buying' : ['vhigh', 'high', 'med', 'low']
                    , 'maint' : ['vhigh', 'high', 'med', 'low']
                    , 'doors' : ['2', '3', '4', '5more']
                    , 'persons' : ['2', '4', 'more']
                    , 'lug_boot' : ['small', 'med', 'big']
                    , 'safety' : ['low', 'med', 'high']}
    train = pd.read_csv('./data/car/train.csv')
    train.columns = columns
    test = pd.read_csv('./data/car/test.csv')
    test.columns = columns

    gain_types = ['IG', 'GI', 'ME']
    depths = range(1,7)
        
    report_df = pd.DataFrame(columns=['gain_type', 'depth', 'train_accuracy', 'test_accuracy'])
    for gt in gain_types:
        for d in depths:
            id3_tree = ID3(train=train, attribute_values = attr_values, label_values = label_values, 
                scoring_method=gt, max_depth=d)

            predictions = id3_tree.Predict(test=train)
            train_acc = np.mean(predictions == train['label']).round(4)
            print(f"gain criteria: {gt}, max depth: {d}, train accuracy: {train_acc}")

            predictions = id3_tree.Predict(test=test)
            test_acc = np.mean(predictions == test['label']).round(4)
            print(f"gain criteria: {gt}, max depth: {d}, test accuracy: {test_acc}")

            elem_df = pd.DataFrame.from_dict({'gain_type': [gt], 'depth': [d], 'train_accuracy': [train_acc], 'test_accuracy': [test_acc]})
            report_df = report_df.append(elem_df)

            print()

        print()
    
    report_df.reset_index(drop=True, inplace=True)
    report_df.to_csv("./output/2bout.csv")

def create_bank_table(replace_unknown):
    label_values = ['yes', 'no']
    columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 
        'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']

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



    train = pd.read_csv('./data/bank/train.csv')
    train.columns = columns
    test = pd.read_csv('./data/bank/test.csv')
    test.columns = columns

    if replace_unknown:
        for key in attr_values.keys():
            if 'unknown' in attr_values[key]:
                attr_values[key].remove('unknown')
                most_common = train[key][train[key] != 'unknown'].mode()[0]
                train.loc[train[key] == 'unknown', key] = most_common
                test.loc[test[key] == 'unknown', key] = most_common

    gain_types = ['IG', 'GI', 'ME']
    depths = range(1,17)

    report_df = pd.DataFrame(columns=['gain_type', 'depth', 'train_accuracy', 'test_accuracy'])
    for gt in gain_types:
        for d in depths:
            id3_tree = ID3(train=train, attribute_values = attr_values, label_values = label_values, 
                scoring_method=gt, max_depth=d)

            predictions = id3_tree.Predict(test=train)
            train_acc = np.mean(predictions == train['label']).round(4)
            print(f"gain criteria: {gt}, max depth: {d}, train accuracy: {train_acc}")

            predictions = id3_tree.Predict(test=test)
            test_acc = np.mean(predictions == test['label']).round(4)
            print(f"gain criteria: {gt}, max depth: {d}, test accuracy: {test_acc}")

            elem_df = pd.DataFrame.from_dict({'gain_type': [gt], 'depth': [d], 'train_accuracy': [train_acc], 'test_accuracy': [test_acc]})
            report_df = report_df.append(elem_df)

            print()

        print()
    
    report_df.reset_index(drop=True, inplace=True)
    if replace_unknown:
        report_df.to_csv("./output/3bout.csv")
    else:
        report_df.to_csv("./output/3aout.csv")



                
# print("Car dataset Output:\n\n")
# create_car_table()
# print("Bank dataset without replacement Output:\n\n")
create_bank_table(False)
# print("Bank dataset with replacement Output:\n\n")
# create_bank_table(True)
