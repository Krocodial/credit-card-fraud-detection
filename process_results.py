import csv
import numpy as np
import matplotlib.pyplot as plt


def process_results():
    # open csv file
    with open('rf_results.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # header

        param_names = ['max_features', 'n_estimators', 'class_weights', 'min_samples_split', 'max_depth']

        metric_names = ['vroc_auc', 'vprecision', 'vrecall', 'vf1', 'vfp', 'vfn',
                        'troc_auc', 'tprecision', 'trecall', 'tf1', 'tfp', 'tfn']

        params = {}
        metrics = {}

        for name in param_names:
            params[name] = []

        for name in metric_names:
            metrics[name] = []

        firstline = True
        for row in reader:
            if firstline:
                print(row)
                firstline = False
                continue

            # First 5 cols: hyperparameters
            for i in range(5):
                params[param_names[i]].append(row[i])

            # next 12: validation + test metrics
            for i in range(12):
                metrics[metric_names[i]].append(float(row[i + 5]))

        # collect data
        #eval_max_features(params['max_features'], metrics, metric_names)
        #eval_class_weights(params['class_weights'], metrics, metric_names)
        eval_and_graph(params['min_samples_split'], metrics, metric_names)


def eval_and_graph(param, metrics, metric_names):
    # Arrange the data for plotting
    lines = []

    values_dict = {}

    for name in metric_names:
        if name not in values_dict.keys():
            values_dict[name] = {}

        for i in range(len(param)):
            param[i] = float(param[i])
            if param[i] not in values_dict[name].keys():
                values_dict[name][param[i]] = []

            values_dict[name][param[i]].append(metrics[name][i])

    for name in metric_names:
        if name == 'vfp' or name == 'vfn' or name == 'tfp' or name == 'tfn':
        #if name != 'vfp' and name != 'vfn' and name != 'tfp' and name != 'tfn':
            continue
        else:
            x = list(values_dict[name].keys())
            y = list(values_dict[name].values())

            y = [sum(item)/len(item) for item in y]

            order = np.argsort(x)
            xs = np.array(x)[order]
            ys = np.array(y)[order]

            line, = plt.plot(xs, ys, label=name)
            lines.append(line)

    plt.legend(handles=lines)
    plt.xlabel("Second Hidden Layer Size")
    plt.ylabel("Score")
    #plt.ylabel("Error")
    plt.show()


def eval_class_weights(class_weights, metrics, metric_names):
    # possible values
    same_weight = {}
    slight_skew = {}
    large_skew = {}

    for i in range(len(class_weights)):
        if class_weights[i] == '{0: 1, 1: 1}':
            for name in metric_names:
                if not name in same_weight.keys():
                    same_weight[name] = []

                same_weight[name].append(float(metrics[name][i]))

        elif class_weights[i] == '{0: 0.75, 1: 1.5}':
            for name in metric_names:
                if not name in slight_skew.keys():
                    slight_skew[name] = []

                slight_skew[name].append(float(metrics[name][i]))

        elif class_weights[i] == '{0: 0.5, 1: 2}':
            for name in metric_names:
                if not name in large_skew.keys():
                    large_skew[name] = []

                large_skew[name].append(float(metrics[name][i]))

    # distill metrics
    print('Same Weight')
    for name in metric_names:
        same_weight_mean = np.array(same_weight[name]).mean()
        print("%s:\t%0.2f" % (name, same_weight_mean))

    print('Slight Skew')
    for name in metric_names:
        slight_skew_mean = np.array(slight_skew[name]).mean()
        print("%s:\t%0.2f" % (name, slight_skew_mean))

    print('Large Skew')
    for name in metric_names:
        large_skew_mean = np.array(large_skew[name]).mean()
        print("%s:\t%0.2f" % (name, large_skew_mean))


def eval_max_features(max_features, metrics, metric_names):
    all_feats = {}
    sqrt = {}
    log2 = {}

    for i in range(len(max_features)):

        if max_features[i] == '':
            for name in metric_names:
                if not name in all_feats.keys():
                    all_feats[name] = []

                all_feats[name].append(float(metrics[name][i]))

        elif max_features[i] == 'sqrt':
            for name in metric_names:
                if not name in sqrt.keys():
                    sqrt[name] = []

                sqrt[name].append(float(metrics[name][i]))

        elif max_features[i] == 'log2':
            for name in metric_names:
                if not name in log2.keys():
                    log2[name] = []

                log2[name].append(float(metrics[name][i]))

    # distill metrics
    print('All features')
    for name in metric_names:
        all_mean = np.array(all_feats[name]).mean()
        print("%s:\t%0.2f" % (name, all_mean))

    print('\nSQRT')
    for name in metric_names:
        sqrt_mean = np.array(sqrt[name]).mean()
        print("%s:\t%0.2f" % (name, sqrt_mean))

    print('\nLog2')
    for name in metric_names:
        log2_mean = np.array(log2[name]).mean()
        print("%s:\t%0.2f" % (name, log2_mean))


def process_results_max_depth():
    # open csv file
    with open('rf_results_max_depth.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)

        metric_names = ['vroc_auc', 'vprecision', 'vrecall', 'vf1', 'vfp', 'vfn',
                        'troc_auc', 'tprecision', 'trecall', 'tf1', 'tfp', 'tfn']

        metrics = {}
        max_depth = []

        for name in metric_names:
            metrics[name] = []

        firstline = True
        for row in reader:
            if firstline:
                print(row)
                firstline = False
                continue

            # First column max depth
            max_depth.append(row[0])

            # next 12: validation + test metrics
            for i in range(12):
                metrics[metric_names[i]].append(float(row[i + 1]))

        # evaluate result
        eval_and_graph(max_depth, metrics, metric_names)


def process_results_max_depth():
    # open csv file
    with open('rf_results_n_estimators.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)

        metric_names = ['vroc_auc', 'vprecision', 'vrecall', 'vf1', 'vfp', 'vfn',
                        'troc_auc', 'tprecision', 'trecall', 'tf1', 'tfp', 'tfn']

        metrics = {}
        num_estimators = []

        for name in metric_names:
            metrics[name] = []

        firstline = True
        for row in reader:
            if firstline:
                print(row)
                firstline = False
                continue

            # First column max depth
            num_estimators.append(row[0])

            # next 12: validation + test metrics
            for i in range(12):
                metrics[metric_names[i]].append(float(row[i + 1]))

        # evaluate result
        eval_and_graph(num_estimators, metrics, metric_names)

def process_results_max_depth():
    # open csv file
    with open('rf_results_n_estimators.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)

        metric_names = ['vroc_auc', 'vprecision', 'vrecall', 'vf1', 'vfp', 'vfn',
                        'troc_auc', 'tprecision', 'trecall', 'tf1', 'tfp', 'tfn']

        metrics = {}
        num_estimators = []

        for name in metric_names:
            metrics[name] = []

        firstline = True
        for row in reader:
            if firstline:
                print(row)
                firstline = False
                continue

            # First column max depth
            num_estimators.append(row[0])

            # next 12: validation + test metrics
            for i in range(12):
                metrics[metric_names[i]].append(float(row[i + 1]))

        # evaluate result
        eval_and_graph(num_estimators, metrics, metric_names)

def process_results_layer_size():
    with open('nn_resultslayer2_nosmote.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)

        metric_names = ['vroc_auc', 'vprecision', 'vrecall', 'vf1', 'vfp', 'vfn',
                        'troc_auc', 'tprecision', 'trecall', 'tf1', 'tfp', 'tfn']

        metrics = {}
        num_estimators = []

        for name in metric_names:
            metrics[name] = []

        firstline = True
        for row in reader:
            if firstline:
                print(row)
                firstline = False
                continue

            # First column alpha
            num_estimators.append(row[0])
            print(row[0])

            # next 12: validation + test metrics
            for i in range(12):
                metrics[metric_names[i]].append(float(row[i + 1]))

        # evaluate result
        eval_and_graph(num_estimators, metrics, metric_names)


if __name__ == "__main__":
    process_results_layer_size()
