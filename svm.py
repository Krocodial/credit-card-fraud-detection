from sklearn import svm
import numpy as np
import preprocessing
import validation


def svm_predict(train, test, preprocessing_type):
    # separate class label (last column)
    train, labels = preprocessing.split_labels(train)

    if preprocessing_type == 'smote':
        train, labels = preprocessing.apply_smote(train, labels)

    # Classifier
    # Class weight parameter: weights positive class more strongly than negative class.
    # class_weight={1: 2.61, 0: 0.383}
    classifier = svm.SVC(kernel='rbf')

    validation.cross_validate(classifier, train, labels)
    classifier.fit(train, labels)

    # test
    test, test_labels = preprocessing.split_labels(test)
    validation.test(classifier, test, test_labels)


def basic():
    train, test = preprocessing.prepare_data()
    train = preprocessing.undersample_negative_class(train, 1000)
    svm_predict(train, test, 'basic')


def smote_test():
    train, test = preprocessing.prepare_data()
    train = preprocessing.undersample_negative_class(train, 5000)
    svm_predict(train, test, 'smote')


def main():
    basic()


if __name__ == '__main__':
    main()
