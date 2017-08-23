import preprocessing
import validation
import pandas as pd
import csv

from sklearn.neural_network import MLPClassifier

def nn_predict(train, labels, test, writer, params):

    hidden_layer_sizes = None
    if params['layer_2'] == 0:
        hidden_layer_sizes = (params['layer_1'],)
    elif params['layer_3'] == 0:
        hidden_layer_sizes = (params['layer_1'], params['layer_2'])
    else:
        hidden_layer_sizes = (params['layer_1'], params['layer_2'], params['layer_3'])

    clf = MLPClassifier(solver='lbfgs', alpha=params['alpha'],
                        hidden_layer_sizes=hidden_layer_sizes, random_state=1)
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

    with open('nn_resultsalpha_nosmote.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # add header
        # vary each parameter of random forest
        num_layers_1 = 20
        num_layers_2 = 0
        num_layers_3 = 0
        for alpha in [1.e0, 1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5]:
            if num_layers_2 == 0 and num_layers_3 != 0:
                continue
            split_train, split_labels = preprocessing.split_labels(train)
            params = {'layer_1': num_layers_1,
                      'layer_2': num_layers_2,
                      'layer_3': num_layers_3,
                      'alpha': alpha}
            nn_predict(split_train, split_labels, test, writer, params)

def main():
    basic()


if __name__ == "__main__":
    main()