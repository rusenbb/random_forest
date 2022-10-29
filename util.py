import numpy as np


def entropy(class_y):
    """
    This method calculates entropy
    Args:
        class_y: list of class labels
    Returns:
        entropy: entropy value

    Example: entropy for [0,0,0,1,1,1] should be 1.
    """
    #  /$$$$$$$$ /$$$$$$ /$$       /$$
    # | $$_____/|_  $$_/| $$      | $$
    # | $$        | $$  | $$      | $$
    # | $$$$$     | $$  | $$      | $$
    # | $$__/     | $$  | $$      | $$
    # | $$        | $$  | $$      | $$
    # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
    # |__/      |______/|________/|________/
    return entropy


def information_gain(previous_y, current_y):
    """
    This method calculates information gain. In this method, use the entropy function you filled
    Args:
        y_before: the distribution of target values before splitting
        y_splitted: the distribution of target values after splitting

    Returns:
        information_gain

    Example: if y_before = [0,0,0,1,1,1] and y_splitted = [[0,0],[0,1,1,1]], information_gain = 0.4691

    """

    info_gain = 0
    #  /$$$$$$$$ /$$$$$$ /$$       /$$
    # | $$_____/|_  $$_/| $$      | $$
    # | $$        | $$  | $$      | $$
    # | $$$$$     | $$  | $$      | $$
    # | $$__/     | $$  | $$      | $$
    # | $$        | $$  | $$      | $$
    # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
    # |__/      |______/|________/|________/
    return info_gain


def split_node(X, y, split_feature, split_value):
    """
    This method implements binary split to your X and y.
    Args:
        X: dataset without target value
        y: target labels
        split_feature: column index of the feature to split on
        split_value: value that is used to split X and y into two parts

    Returns:
        X_left: X values for left subtree
        X_right: X values for right subtree
        y_left: y values of X values in the left subtree
        y_right: y values of X values in the right subtree

    Notes:  Implement binary split.
            You can use mean for split_value or you can try different split_value values for better Random Forest results
            Assume you are only dealing with numerical features. You can ignore the case there are categorical features.
            Example:
                Divide X into two list.
                X_left: where values are <= split_value.
                X_right: where values are > split_value.
    """

    X_left = []
    X_right = []

    y_left = []
    y_right = []
    #  /$$$$$$$$ /$$$$$$ /$$       /$$
    # | $$_____/|_  $$_/| $$      | $$
    # | $$        | $$  | $$      | $$
    # | $$$$$     | $$  | $$      | $$
    # | $$__/     | $$  | $$      | $$
    # | $$        | $$  | $$      | $$
    # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
    # |__/      |______/|________/|________/
    return X_left, X_right, y_left, y_right


def confusion_matrix_(y_predicted, y):
    """
    Args:
        y_predicted: predicted number of features
        y: your true labels

    Returns:
        confusion_matrix: with shape (2, 2)

    """
    confusion_matrix = np.zeros((2, 2))
    #  /$$$$$$$$ /$$$$$$ /$$       /$$
    # | $$_____/|_  $$_/| $$      | $$
    # | $$        | $$  | $$      | $$
    # | $$$$$     | $$  | $$      | $$
    # | $$__/     | $$  | $$      | $$
    # | $$        | $$  | $$      | $$
    # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
    # |__/      |______/|________/|________/

    return confusion_matrix


def eval_metrics(conf_matrix):
    """
    Args:
        conf_matrix: Use confusion matrix you calculated

    Returns:
        accuracy, recall, precision, f1_score

    """
    accuracy, recall, precision, f1_score = 0, 0, 0, 0
    #  /$$$$$$$$ /$$$$$$ /$$       /$$
    # | $$_____/|_  $$_/| $$      | $$
    # | $$        | $$  | $$      | $$
    # | $$$$$     | $$  | $$      | $$
    # | $$__/     | $$  | $$      | $$
    # | $$        | $$  | $$      | $$
    # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
    # |__/      |______/|________/|________/

    return accuracy, recall, precision, f1_score
