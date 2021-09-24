import numpy as np
import pandas as pd
import math


class ID3:

    class Node:
        def __init__(self, branch_value=None, children=[], split_attribute=None, depth=0, label=None):
            self.branch_value = branch_value
            self.children = []
            self.split_attribute = split_attribute
            self.label = label
            self.depth = depth


    def __init__(self, train, attribute_values, label_values, scoring_method, max_depth):
        self.data = train.copy()
        self.attribute_values = attribute_values.copy()
        self.label_values = label_values.copy()

        if scoring_method.upper() == "ME":
            self.score = self.me_gain
        elif scoring_method.upper() == "IG":
            self.score = self.ig_gain
        elif scoring_method.upper() == "GI":
            self.score = self.gi_gain
        else:
            self.score = self.ig_gain

        numeric_medians = train.select_dtypes(include=np.number).median()

        for attribute, median in numeric_medians.iteritems():
            self.attribute_values[attribute] = median


        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.ordered_split_attributes = {}
        self.tree = self.create_tree(self.data, list(self.data.columns[:-1]))

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


    def best_split_attribute(self, data, attributes):
        scores = [self.score(data[[a, 'label']], a).round(4) for a in attributes]
        max_score_idx = np.argmax(scores)
        return attributes[max_score_idx]
        

    def create_tree(self, data, attributes, branch_value='', depth=1):
        root = self.Node(branch_value=branch_value)

        label_arr = data['label'].to_numpy()
        if np.all(label_arr == label_arr[0]):
            root.label = label_arr[0]
            return root

        if len(attributes) == 0 or depth > self.max_depth:
            root.label = data['label'].mode()[0]
            return root

        best_attribute = self.best_split_attribute(data, attributes)

        self.ordered_split_attributes[best_attribute] = None

        root.split_attribute = best_attribute
        if data[best_attribute].dtype != object:
            subsets = (data[data[best_attribute] > self.attribute_values[best_attribute]], 
                        data[data[best_attribute] <= self.attribute_values[best_attribute]])
            attr_values = ['above', 'below_equal']
        else:
            subsets = [data[data[best_attribute] == attr_val] for attr_val in self.attribute_values[best_attribute]]
            attr_values = self.attribute_values[best_attribute]
                

        for subset, attr_val in zip(subsets, attr_values):
            if subset.shape[0] == 0:
                root.children.append(self.Node(branch_value=attr_val, label=data['label'].mode()[0])) 
            else:
                root.children.append(self.create_tree(subset, [a for a in attributes if a != best_attribute], attr_val, depth + 1))

        return root


    def Predict(self, test):
        results = []
        for row in test.itertuples(index=False):
            node = self.tree

            while node.split_attribute:
                row_attr = getattr(row, node.split_attribute)
                if type(row_attr) != str:
                    row_attr = 'above' if row_attr > self.attribute_values[node.split_attribute] else 'below_equal'

                for child in node.children:
                    if child.branch_value == row_attr:
                        node = child
                        break

            results.append(node.label)

        return results



