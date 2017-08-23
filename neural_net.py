import preprocessing
import validation
import pandas as pd
import csv

from sklearn.neural_network import MLPClassifier

def nn_predict(train, labels, test, writer):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(20,), random_state=1)
    vresult = validation.cross_validate(clf, train, labels)

    clf.fit(train, labels)

    # test
    test_labels = test[['Class']]
    test_labels = test_labels.values.ravel()
    test = test.drop('Class', 1)
    tresult = validation.test(clf, test, test_labels)
    results = list()
    results.append(vresult['roc_auc'])
    results.append(vresult['precision'])
    results.append(vresult['recall'])
    results.append(vresult['f1'])
    results.append(vresult['fp'])
    results.append(vresult['fn'])

    results.append(tresult['roc_auc'])
    results.append(tresult['precision'])
    results.append(tresult['recall'])
    results.append(tresult['f1'])
    results.append(tresult['fp'])
    results.append(tresult['fn'])

    writer.writerow(results)

def basic():
    train, test = preprocessing.prepare_data(True)
    train = train.drop('Amount', axis=1)
    test = test.drop('Amount', axis=1)

    with open('nn_results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        split_train, split_labels = preprocessing.split_labels(train)
        nn_predict(split_train, split_labels, test, writer)

        split_train, split_labels = preprocessing.apply_smote(split_train, split_labels)
        nn_predict(split_train, split_labels, test, writer)


def main():
    basic()


if __name__ == "__main__":
    main()