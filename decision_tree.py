from util import entropy, information_gain, split_node
import numpy as np


class DecisionTree(object):
    def __init__(self):
        self.tree = {} # you can use different data structure to build your tree

    def train(self, X, y):
        """
        This method trains decision tree (trains = construct = build)
        Args:
            X: data excluded target feature
            y: target feature

        Returns:

        NOTES:  You can add more parameter to build your algorithm, if necessary.
                You will have to use the functions from util.py
                Construct your tree using dictionary / or you can choose any other data structure.
                Each key should represent a property of your tree and value is corresponding value for your key
        IMPORTANT:  ADD RANDOMNESS
                    You should add randomness to your decision tree for random forest
                    At each node: select random 5 features from feature list (22 feature) and
                    compare the information gain of only 5 randomly selected features to select splitting attribute with the highest information gain
                    The selected random features should change at every node choice
        Example: you can think each node as Node object and node object should have some properties.
                A node should have split_value and split_attribute properties that give us the information of that node.
                (Below example is just an example, each tree should have more properties)
                Like: tree["split_value"] = split_value_you_find
                      tree["split_attribute"] = split_feature_with_highest_information_gain
        """

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        pass

    def classify(self, record):
        """
        This method classifies the record using the tree you build and returns the predicted la
        Args:
            record: each data point

        Returns:
            predicted_label

        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/

        pass
