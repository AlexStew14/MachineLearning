import numpy as np
import pandas as pd
import math
import random

class ID3:

    class Node:
        def __init__(self, branch_value=None, children=[], split_attribute=None, depth=0, label=None):
            self.branch_value = branch_value
            self.children = []
            self.split_attribute = split_attribute
            self.label = label
            self.depth = depth


    def __init__(self, train, attribute_values, label_values, scoring_method, max_depth, weights=None, stump=False, attribute_sample=None):
        self.data = train.copy()        
        self.attribute_values = attribute_values.copy()
        self.label_values = label_values.copy()        

        self.stump = stump
        
        if scoring_method.upper() == "ME":
            self.score = self.me_gain
        elif scoring_method.upper() == "IG":
            self.score = self.ig_gain
        elif scoring_method.upper() == "GI":
            self.score = self.gi_gain
        else:
            self.score = self.ig_gain

        if stump:
            self.score = self.ig_gain_adaboost

        numeric_medians = train.select_dtypes(include=np.number).median()

        for attribute, median in numeric_medians.iteritems():
            if attribute != 'label':
                self.attribute_values[attribute] = [0,1]
                self.data[attribute] = (self.data[attribute] >= median).astype(int)

        self.max_depth = max_depth if max_depth is not None else float('inf')

        self.attribute_sample = attribute_sample

        self.weights = None
        if self.stump:
            self.max_depth = 2
            self.weights = weights.copy()

        self.ordered_split_attributes = {}
        self.tree = self.create_tree(self.data, list(self.data.columns[:-1]), weights=self.weights)


    def ig_gain(self, data, column_name):
        original_entropy = 0
        for p in data['label'].value_counts(normalize=True, sort=False):
            original_entropy -= p * math.log2(p + 1e-8)

        counts = data[column_name].value_counts(normalize=True, sort=False)
        for a_val, weight in zip(counts.index, counts.values):
            sv = data[data[column_name] == a_val]
            probabilities = sv['label'].value_counts(normalize=True, sort=False).values
            original_entropy -= -(np.sum([p * np.log2(p + 1e-8) for p in probabilities])) * weight

        return original_entropy


    def ig_gain_adaboost(self, data, column_name, weights):
        original_entropy = 0

        for l_val in pd.unique(data['label']):
            p = np.sum(weights[data['label']==l_val])
            original_entropy -= p * math.log2(p + 1e-8)

        for a_val in pd.unique(data[column_name]):
            sv = data[data[column_name] == a_val]
            sv_weights = weights[data[column_name] == a_val]
            a_val_weight = np.sum(sv_weights)
            for l_val in pd.unique(sv['label']):
                a_val_l_val_weights = sv_weights[sv['label']==l_val]
                p = np.sum(a_val_l_val_weights) / a_val_weight
                original_entropy += (p * np.log2(p + 1e-8)) * a_val_weight
                
        return original_entropy


    def me_gain(self, data, column_name):
        original_me = np.min(data['label'].value_counts(normalize=True, sort=False).values)

        counts = data[column_name].value_counts(normalize=True, sort=False)
        for a_val, weight in zip(counts.index, counts.values):
            subset = data[data[column_name] == a_val]
            sv_me = 1 - np.max(subset['label'].value_counts(normalize=True, sort=False).values)
            original_me -= sv_me * weight
                
        return original_me


    def gi_gain(self, data, column_name):
        original_gi = 1 - np.sum(np.square(data['label'].value_counts(normalize=True, sort=False).values))

        counts = data[column_name].value_counts(normalize=True, sort=False)
        for a_val, weight in zip(counts.index, counts.values):
            subset = data[data[column_name] == a_val]
            sv_probabilities = subset['label'].value_counts(normalize=True, sort=False).values
            subset_gi = 1 - np.sum(np.square(sv_probabilities))
            original_gi -= subset_gi * weight
                
        return original_gi


    def best_split_attribute(self, data, attributes, weights):
        if weights is None:
            scores = [self.score(data[[a, 'label']], a).round(4) for a in attributes]
        elif self.attribute_sample is not None:
            scores = [self.score(data[[a, 'label']], a).round(4) for a in random.sample(attributes, self.attribute_sample)]
        else:
            scores = [self.score(data[[a, 'label']], a, weights).round(4) for a in attributes]

        max_score_idx = np.argmax(scores)
        return attributes[max_score_idx]
        

    def create_tree(self, data, attributes, branch_value='', depth=1, weights=None):
        root = self.Node(branch_value=branch_value)

        label_arr = data['label'].to_numpy()
        if np.all(label_arr == label_arr[0]):
            root.label = label_arr[0]
            return root

        if len(attributes) == 0 or depth >= self.max_depth:
            if self.stump:
                max_weight = -np.inf
                root_label = None
                for val in pd.unique(data.label):
                    weight_sum = np.sum(weights[data.label == val])
                    if weight_sum > max_weight:
                        max_weight =weight_sum
                        root_label = val

                root.label = root_label
            else:
                root.label = data['label'].mode()[0]
            return root

        best_attribute = self.best_split_attribute(data, attributes, weights)

        self.ordered_split_attributes[best_attribute] = None

        root.split_attribute = best_attribute
                
        subsets = [data[data[best_attribute] == attr_val] for 
                            attr_val in self.attribute_values[best_attribute]]

        weights_subsets = None
        if self.stump:
            weights_subsets = [weights[data[best_attribute] == attr_val] for 
                                attr_val in self.attribute_values[best_attribute]]    

        attr_values = self.attribute_values[best_attribute]
                

        i = 0
        for subset, attr_val in zip(subsets, attr_values):
            if subset.shape[0] == 0:
                root.children.append(self.Node(branch_value=attr_val, label=data['label'].mode()[0])) 
            else:
                root.children.append(self.create_tree(subset, [a for a in attributes if a != best_attribute], 
                                    attr_val, depth + 1, None if not self.stump else weights_subsets[i]))
            i += 1

        return root


    def Predict(self, test):
        test = test.copy()
        numeric_medians = test.select_dtypes(include=np.number).median()

        for attribute, median in numeric_medians.iteritems():
            test[attribute] = (test[attribute] >= median).astype(int)

        results = []
        for row in test.itertuples(index=False):
            node = self.tree

            while node.split_attribute:
                row_attr = getattr(row, node.split_attribute)

                for child in node.children:
                    if child.branch_value == row_attr:
                        node = child
                        break

            results.append(node.label)

        return results


