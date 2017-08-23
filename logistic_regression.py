from sklearn import linear_model
import preprocessing
import validation


def lr_predict(train, test, preprocessing_type):
    # separate class label (last column)
    train, labels = preprocessing.split_labels(train)

    if preprocessing_type == 'smote':
        train, labels = preprocessing.apply_smote(train, labels)

    classifier = linear_model.LogisticRegression()


    validation.cross_validate(classifier, train, labels)
    classifier.fit(train, labels)

    # test
    test, test_labels = preprocessing.split_labels(test)
    validation.test(classifier, test, test_labels)


def multiple_balanced_sets():
    train, test = preprocessing.prepare_data()
    train_list = preprocessing.multiple_balanced_samples(train, 5)
    # separate class label (last column)
    for i in range(5):
        train, labels = preprocessing.split_labels(train_list[i])
        classifier = linear_model.LogisticRegression()
        validation.cross_validate_set(classifier, train, labels)

    validation.cross_validate(classifier, train, labels)
    classifier.fit(train, labels)

    # test
    test, test_labels = preprocessing.split_labels(test)
    validation.test(classifier, test, test_labels)

def basic():
    train, test = preprocessing.prepare_data()
    lr_predict(train, test, 'basic')


def smote_test():
    train, test = preprocessing.prepare_data()
    lr_predict(train, test, 'smote')


def main():
    multiple_balanced_sets()


if __name__ == '__main__':
    main()
