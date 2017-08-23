from sklearn import model_selection
import pandas as pd
import math
from imblearn import over_sampling


# Prepares data set, undersampling the majority (negative) class
def prepare_data(normalize = False):
    # get dataset
    dataframe = pd.read_csv("creditcard.csv")
    print("Loaded dataset")

    if normalize:
        dataframe['Time'] = dataframe['Time'].apply(lambda x: (float(x) % 86400) / 86400.0)
        dataframe['Amount'] = dataframe['Amount'].apply(lambda x: math.log10(max([float(x), 1])))

        cols = [ 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
                 'V7', 'V8', 'V9', 'V10', 'V11', 'V12',
                 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                 'V19', 'V20', 'V21', 'V22', 'V23', 'V24',
                 'V25', 'V26', 'V27', 'V28' ]
        dataframe[cols] = dataframe[cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Split into training and test
    # Important to have minority class represented in both train and test,
    # so we split the dataset into classes first.
    pos = dataframe.loc[dataframe['Class'] == 1]
    neg = dataframe.loc[dataframe['Class'] == 0]

    pos_train, pos_test = model_selection.train_test_split(pos, test_size=0.2, random_state=0)
    neg_train, neg_test = model_selection.train_test_split(neg, test_size=0.2, random_state=0)

    train = pd.concat([pos_train, neg_train])
    test = pd.concat([pos_test, neg_test])

    # Randomize order of samples.
    train = train.sample(frac=1)
    train = train.reset_index(drop=True)

    test = test.sample(frac=1)
    test = test.reset_index(drop=True)

    return train, test


# Create several balanced training data sets
def multiple_balanced_samples(train, sets):
    pos = train.loc[train['Class'] == 1]
    neg = train.loc[train['Class'] == 0]

    # number of positive samples
    pos_count = len(pos)
    neg_count = len(neg)
    samples = []

    for i in range(sets):
        neg_small_set = neg.sample(frac=(pos_count/neg_count))
        samples.append(pd.concat([pos, neg_small_set]))

        # randomize order
        samples[i] = samples[i].sample(frac=1)
        samples[i] = samples[i].reset_index(drop=True)

    return samples


# Prepares data set, undersampling the majority (negative) class
def undersample_negative_class(train, neg_size):

    neg_train = train.loc[train['Class'] == 0]
    pos_train = train.loc[train['Class'] == 1]

    neg_train = neg_train.head(neg_size)

    train = pd.concat([pos_train, neg_train])
    return train


def split_labels(train):
    # separate class label (last column)
    labels = train[['Class']]
    data = train.drop('Class', 1)
    labels = labels.values.ravel()

    return data, labels


# Preprocessing techniques for unbalanced datasets
def apply_smote(data, labels):
    smote = over_sampling.SMOTE()
    # Fits the data set and creates new samples for the minority class.
    return smote.fit_sample(data, labels)


if __name__ == "__main__":
    train, test = prepare_data()
    neg_train = train.loc[train['Class'] == 0]
    pos_train = train.loc[train['Class'] == 1]

    print("neg_train: %d, pos_train: %d" % (len(neg_train), len(pos_train)))

    sets = multiple_balanced_samples(train, 5)

    for set in sets:
        neg_set = set.loc[set['Class'] == 0]
        pos_set = set.loc[set['Class'] == 1]
        print("split_neg: %d, split_pos: %d" % (len(neg_set), len(pos_set)))

    # data, labels = apply_smote(train)
    # neg_labels = [x for x in labels if x == 0]
    # pos_labels = [x for x in labels if x == 1]
    # print("overall: %d, neg_train: %d, pos_train: %d" % (len (labels), len(neg_labels), len(pos_labels)))


