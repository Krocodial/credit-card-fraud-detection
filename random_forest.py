import preprocessing
import validation
from sklearn import ensemble
from sklearn import model_selection
import csv


def rf_predict(train, test, preprocessing_type, results_file):
    # separate class label
    train, labels = preprocessing.split_labels(train)

    if preprocessing_type == 'smote':
        train, labels = preprocessing.apply_smote(train, labels)

    classifier = ensemble.RandomForestClassifier(class_weight={0: 0.75, 1: 1.5},
                                                 min_samples_split=40,
                                                 n_estimators=15)

    classifier.fit(train, labels)

    vresult = validation.cross_validate(classifier, train, labels)

    # test
    test, test_labels = preprocessing.split_labels(test)
    tresult = validation.test(classifier, test, test_labels)

    # save results
    results = []
    results.append("low_skew (0=0.75, 1=1.5)")
    results.append(40)
    results.append(15)
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

    results_file.writerow(results)


def basic():
    train, test = preprocessing.prepare_data()

    with open('final_results_basic.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # add header
        writer.writerow(['class_weight', 'min_samples_split', 'n_estimators',
                         'vroc_auc', 'vprecision', 'vrecall', 'vf1', 'vfp', 'vfn',
                         'troc_auc', 'tprecision', 'trecall', 'tf1', 'tfp', 'tfn'])
        # vary each parameter of random forest
        rf_predict(train, test, 'basic', writer)


def smote_test():
    train, test = preprocessing.prepare_data()

    with open('final_results_smote.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # add header
        writer.writerow(['max_depth', 'n_estimators', 'min_samples_split', 'class_weight', 'max_features'
                         'vroc_auc', 'vprecision', 'vrecall', 'vf1', 'vfp', 'vfn',
                         'troc_auc', 'tprecision', 'trecall', 'tf1', 'tfp', 'tfn'])

        rf_predict(train, test, 'smote', writer)


def main():
    basic()
    smote_test()


if __name__ == '__main__':
    main()