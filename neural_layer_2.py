import preprocessing
import validation
import pandas as pd
import csv

from sklearn.neural_network import MLPClassifier

def nn_predict(train, labels, test, writer, params):

    clf = MLPClassifier(solver='lbfgs', alpha=1.e-3,
                        hidden_layer_sizes=(20,params['layer_2']), random_state=1)
    vresult = validation.cross_validate(clf, train, labels)

    clf.fit(train, labels)

    # test
    test_labels = test[['Class']]
    test_labels = test_labels.values.ravel()
    test = test.drop('Class', 1)
    tresult = validation.test(clf, test, test_labels)

    results = list(params.values())
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

    with open('nn_resultslayer2_nosmote.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # add header
        # vary each parameter of random forest
        num_layers_1 = 20
        num_layers_2 = 0
        num_layers_3 = 0
        alpha = 1.e-3
        for num_layers_2 in range(1, 30):
            if num_layers_2 == 0 and num_layers_3 != 0:
                continue
            split_train, split_labels = preprocessing.split_labels(train)
            params = {'layer_2': num_layers_2}
            nn_predict(split_train, split_labels, test, writer, params)

def main():
    basic()


if __name__ == "__main__":
    main()