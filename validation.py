import sklearn
from sklearn import model_selection
from sklearn import metrics
import preprocessing


def false_pos(classifier, X, y):
    pred_y = classifier.predict(X)
    return metrics.confusion_matrix(y, pred_y)[0][1]


def false_neg(classifer, X, y):
    pred_y = classifer.predict(X)
    return metrics.confusion_matrix(y, pred_y)[1][0]


def cross_validate(classifier, train, labels):
    print("Validation")
    # Cross-validation step
    # Use stratified k-fold selection to ensure positive samples are represented
    skf = model_selection.StratifiedKFold(n_splits=5, random_state=0)
    scoring = ['roc_auc', 'precision', 'recall', 'f1']
    scores = model_selection.cross_validate(classifier, train, labels, cv=skf, scoring=scoring)

    print("roc_auc:\t%0.2f (+/- %0.2f)" % (scores['test_roc_auc'].mean(), scores['test_roc_auc'].std() * 2))
    print("precision:\t%0.2f (+/- %0.2f)" % (scores['test_precision'].mean(), scores['test_precision'].std() * 2))
    print("recall:\t%0.2f (+/- %0.2f)" % (scores['test_recall'].mean(), scores['test_recall'].std() * 2))
    print("f1:\t%0.2f (+/- %0.2f)\n" % (scores['test_f1'].mean(), scores['test_f1'].std() * 2))

    false_pos_scores = model_selection.cross_val_score(classifier, train, labels, cv=skf, scoring=false_pos)
    false_neg_scores = model_selection.cross_val_score(classifier, train, labels, cv=skf, scoring=false_neg)

    print("false positives:\t%0.2f (+/- %0.2f)" % (false_pos_scores.mean(), false_pos_scores.std() * 2))
    print("false negatives:\t%0.2f (+/- %0.2f)" % (false_neg_scores.mean(), false_neg_scores.std() * 2))

    return {'roc_auc': scores['test_roc_auc'].mean(), 'precision': scores['test_precision'].mean(),
            'recall': scores['test_recall'].mean(), 'f1': scores['test_f1'].mean(), 'fp': false_pos_scores.mean(),
            'fn': false_neg_scores.mean()}


def test(classifier, test_data, test_labels):

    print("Test")
    scores = classifier.predict(test_data)

    roc_auc = metrics.roc_auc_score(test_labels, scores)
    precision = metrics.precision_score(test_labels, scores)
    recall = metrics.recall_score(test_labels, scores)
    f1 = metrics.f1_score(test_labels, scores)
    # evaluate metrics on test data
    print("roc_auc:\t%0.2f" % roc_auc)
    print("precision:\t%0.2f" % precision)
    print("recall:\t%0.2f" % recall)
    print("f1:\t%0.2f" % f1)

    # Evaluate confusion matrix
    print("Confusion matrix")
    matrix = metrics.confusion_matrix(test_labels, scores)
    print(matrix)

    return {'roc_auc': roc_auc, 'precision': precision,
            'recall': recall, 'f1': f1, 'fp': matrix[0][1],
            'fn': matrix[1][0]}
