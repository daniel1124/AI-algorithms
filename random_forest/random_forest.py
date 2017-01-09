import numpy as np
from collections import Counter

class DecisionNode():
    """Class to represent a single node in
    a decision tree."""

    def __init__(self, left, right, decision_function,class_label=None):
        """Create a node with a left child, right child,
        decision function and optional class label
        for leaf nodes."""
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Return on a label if node is leaf,
        or pass the decision down to the node's
        left/right child (depending on decision
        function)."""
        if self.class_label is not None:
            return self.class_label
        elif self.decision_function(feature):
            return self.left.decide(feature)
        else:
            return self.right.decide(feature)

def build_decision_tree():
    """Create decision tree
    capable of handling the provided 
    data."""
    decision_tree_root = DecisionNode(
        DecisionNode(None, None, None, class_label=1),
        DecisionNode(
            DecisionNode(
                DecisionNode(None, None, None, class_label=1),
                DecisionNode(None, None, None, class_label=0),
                lambda x: x[3] == 1),
            DecisionNode(
                DecisionNode(None, None, None, class_label=0),
                DecisionNode(None, None, None, class_label=1),
                lambda x: x[3] == 1),
            lambda x: x[2] == 1),
        lambda x: x[0] == 1
    )

    return decision_tree_root

def confusion_matrix(classifier_output, true_labels):
    '''[[true_positive, false_negative], [false_positive, true_negative]]'''
    pred = np.array(classifier_output)
    true = np.array(true_labels)

    return [[sum((pred == 1) & (true == 1)), sum((pred == 0) & (true == 1))],
            [sum((pred == 1) & (true == 0)), sum((pred == 0) & (true == 0))]]


def precision(classifier_output, true_labels):
    '''precision is measured as: true_positive/ (true_positive + false_positive)'''
    pred = np.array(classifier_output)
    true = np.array(true_labels)
    
    return 1.0 * sum((pred == 1) & (true == 1)) / sum(pred == 1)
    
def recall(classifier_output, true_labels):
    '''recall is measured as: true_positive/ (true_positive + false_negative)'''
    pred = np.array(classifier_output)
    true = np.array(true_labels)
    
    return 1.0 * sum((pred == 1) & (true == 1)) / sum(true == 1)
    
def accuracy(classifier_output, true_labels):
    '''accuracy is measured as:  correct_classifications / total_number_examples'''
    pred = np.array(classifier_output)
    true = np.array(true_labels)
    
    return 1.0 * sum(pred == true) / len(true)

def entropy(class_vector):
    """Compute the entropy for a list
    of classes (given as either 0 or 1)."""
    v = np.array(class_vector)
    ct0, ct1 = sum(v == 0), sum(v == 1)
    if ct0 == 0 or ct1 == 0:
        return 0
    n = len(v)        
    return -1.0 / n * (ct0 * np.log2(ct0) + ct1 * np.log2(ct1)) + np.log2(n)
    
def information_gain(previous_classes, current_classes ):
    """Compute the information gain between the
    previous and current classes (a list of 
    lists where each list has 0 and 1 values)."""
    h1 = entropy(previous_classes)
    h2 = 0
    for c in current_classes:
        h2 += 1.0 * len(c) / len(previous_classes) * entropy(c)
    return h1 - h2

class DecisionTree():
    """Class for automatic tree-building
    and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with an empty root
        and the specified depth limit."""
        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__()."""
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):  
        """Implement the above algorithm to build
        the decision tree using the given features and
        classes to build the decision functions."""
        if len(np.unique(classes)) == 1 or depth == self.depth_limit:
            return DecisionNode(None, None, None, np.argmax(np.bincount(classes)))
        
        y = np.array(classes)
        
        n, m = features.shape
        max_info_gain, best_f, best_cut = float('-Inf'), 0, 0
        for i in range(m):
            x = features[:, i]
            xuniq = np.unique(x) # already sorted
            
            if len(xuniq) > 1:
                cutoffs = ((xuniq[:-1] + xuniq[1:]) / 2.0)[::np.max([1, len(xuniq) / 10])]
            else:
                cutoffs = [np.max(x)]
            for cut in cutoffs:
                mask = x <= cut
                info_gain = information_gain(y, [y[mask], y[~mask]])
                if info_gain > max_info_gain:
                    max_info_gain, best_f, best_cut = info_gain, i, cut
        
        left_mask = features[:, best_f] <= best_cut
        
        if sum(left_mask) == len(features) or sum(left_mask) == 0:
            return DecisionNode(None, None, None, np.argmax(np.bincount(y)))
        
        left = self.__build_tree__(features[left_mask], y[left_mask], depth + 1)
        right = self.__build_tree__(features[~left_mask], y[~left_mask], depth + 1)

        return DecisionNode(left, right, lambda x: x[best_f] <= best_cut)
        
    def classify(self, features):
        """Use the fitted tree to 
        classify a list of examples. 
        Return a list of class labels."""
        class_labels = []

        for f in features:            
            class_labels.append(self.root.decide(f))

        return class_labels

def load_csv(data_file_path, class_index=-1):
    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r ])
    classes= map(int,  out[:, class_index])
    features = out[:, :class_index]
    return features, classes

def generate_k_folds(dataset, k):
    #This method returns a list of folds,
    # where each fold is a tuple like (training_set, test_set)
    # where each set is a tuple like (examples, classes)
    features, classes = dataset
    features = np.array(features)
    n = len(classes)
    idx = range(n)
    np.random.shuffle(idx)
    features_ = features[idx]
    classes_ = np.array(classes)[idx]
    folds = []
    size = n / k
    for i in range(k):
        xtest = features_[i * size: (i + 1) * size] 
        ytest = classes_[i * size: (i + 1) * size]
        xtrain = np.concatenate((features_[:i * size], features_[(i + 1) * size:]))
        ytrain = np.concatenate((classes_[:i * size], classes_[(i + 1) * size:]))
        folds.append(((xtrain, ytrain), (xtest, ytest)))
    return folds


class RandomForest():
    """Class for random forest
    classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate, attr_subsample_rate):
        """Create a random forest with a fixed 
        number of trees, depth limit, example
        sub-sample rate and attribute sub-sample
        rate."""
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of 
        decision trees."""
        self.col_idxs = []
        for i in range(self.num_trees):
            tree = DecisionTree(self.depth_limit)
            
            n, m = features.shape
            row_idx = np.random.choice(range(n), int(self.example_subsample_rate * n), True)
            col_idx = np.random.choice(range(m), int(self.attr_subsample_rate * m), False)
            self.col_idxs.append(col_idx)

            features_ = features[row_idx][:, col_idx]
            classes_ = np.array(classes)[row_idx]
                        
            tree.fit(features_, classes_)
            self.trees.append(tree) 

    def classify(self, features):
        """Classify a list of features based
        on the trained random forest."""
        res = []
        for i in range(len(self.trees)):
            tree = self.trees[i]
            features_ = features[:, self.col_idxs[i]]
            res.append(tree.classify(features_))
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), 0, res).tolist()

class CustomClassifier():
    
    def __init__(self, num_trees=20, depth_limit=float('Inf'), example_subsample_rate=1, attr_subsample_rate=0.13):
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        
    def fit(self, features, classes):
        # fit model to the provided features
        self.col_idxs = []
        features = np.array(features)
        classes = np.array(classes)
        for i in range(self.num_trees):
            tree = DecisionTree(self.depth_limit)
            
            n, m = features.shape
            row_idx = np.random.choice(range(n), int(self.example_subsample_rate * n), True)
            col_idx = np.random.choice(range(m), int(self.attr_subsample_rate * m), False)
            self.col_idxs.append(col_idx)

            features_ = features[row_idx][:, col_idx]
            classes_ = classes[row_idx]
                        
            tree.fit(features_, classes_)
            self.trees.append(tree)
        
    def classify(self, features):
        # classify each feature in features as either 0 or 1.
        res = []
        features = np.array(features)
        for i in range(len(self.trees)):
            tree = self.trees[i]
            features_ = features[:, self.col_idxs[i]]
            res.append(tree.classify(features_))
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), 0, res).tolist()


