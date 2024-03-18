import pandas as pd
import numpy as np
import copy
import csv
import pandas as pd

def uniques_and_counts(data, name='anonymous'):
    print(name)
    uniques, counts = np.unique(data, return_counts=True)
    # print(list(zip(uniques, counts)))
    return uniques, counts

def distribution_test(data, sampling='over-sampling'):
    for each in data:
        train = np.array(each[0])
        val = np.array(each[1])
        test = np.array(each[2])
        train_id, train_label = train[:, 0], train[:, 1]
        val_id, val_label = val[:, 0], val[:, 1]
        test_id, test_label = test[:, 0], test[:, 1]

        assert np.sum(np.in1d(train_id, val_id)) == 0
        assert np.sum(np.in1d(train_id, test_id)) == 0

        train_id_uniques, train_id_counts = uniques_and_counts(train_id, 'train id')
        _, train_label_counts = uniques_and_counts(train_label, 'train label')
        if sampling != 'no-sampling':
            assert train_label_counts[0] == train_label_counts[1]
        if sampling == 'under-sampling':
            assert np.sum(train_id_counts) == len(train_id_counts)
        res = np.unique(train_id_counts, return_counts=True)
        print(list(zip(res[0], res[1])))

        val_id_uniques, val_id_counts = uniques_and_counts(val_id, 'val id')
        _, val_label_counts = uniques_and_counts(val_label, 'val label')
        print(val_label_counts[0], val_label_counts[1])
        assert val_label_counts[0] != val_label_counts[1]
        assert np.sum(val_id_counts) == val_id_counts.shape[0]

        test_id_uniques, test_id_counts = uniques_and_counts(test_id, 'test id')
        _, test_label_counts = uniques_and_counts(test_label, 'test label')
        assert test_label_counts[0] != test_label_counts[1]
        assert np.sum(test_id_counts) == test_id_counts.shape[0]

        if sampling != 'under-sampling':
            assert train_id_uniques.shape[0] + val_id_uniques.shape[0] + test_id_uniques.shape[0] == 911


def unique_test(data, name='anonymous'):
    data = np.array(data)
    ids = data[:, 0]
    label = data[:, 1]
    ids_unique, ids_count = np.unique(ids, return_counts=True)
    label_unique, label_count = np.unique(label, return_counts=True)
    ids_map = list(zip(ids_unique, ids_count))
    label_map = list(zip(label_unique, label_count))
    print('===========================')
    print(name)
    print('--------- ids_map ---------')
    print(ids_map)
    print('-------- label_map --------')
    print(label_map)
    print('===========================')

def folds_distributer(data, folds=5, sampling='over-sampling'):
    # over-sampling | under-sampling
    if sampling not in ('over-sampling', 'under-sampling', 'no-sampling'):
        raise

    distributions = [
        [[], [], []],
        [[], [], []],
        [[], [], []],
        [[], [], []],
        [[], [], []]
    ]

    # vals, counts = np.unique(data, return_counts=True)
    data_size = data.shape[0]
    id = list(range(1, data_size + 1))
    data = np.column_stack([id, data])
    np.random.shuffle(data)

    positives = np.empty((0, 2))
    negatives = np.empty((0, 2))
    for i in data:
        if i[-1] == 1:
            positives = np.vstack([positives, i])
        elif i[-1] == 0:
            negatives = np.vstack([negatives, i])
        else:
            raise

    p_fold = [[], [], [], [], []]
    n_fold = [[], [], [], [], []]
    under_p_fold = [[], [], [], [], []]
    under_n_fold = [[], [], [], [], []]

    if positives.shape[0] > negatives.shape[0]:
        for i, item in enumerate(positives):
            fold = i % folds

            p_fold[fold].append(item)

            if i < negatives.shape[0]:
                n_fold[fold].append(negatives[i])
        for i, item in enumerate(negatives):
            fold = i % folds

            under_n_fold[fold].append(item)
            under_p_fold[fold].append(positives[i])
    else:
        for i, item in enumerate(negatives):
            fold = i % folds

            n_fold[fold].append(item)

            if i < positives.shape[0]:
                p_fold[fold].append(positives[i])
        for i, item in enumerate(positives):
            fold = i % folds

            under_p_fold[fold].append(item)
            under_n_fold[fold].append(negatives[i])

    for i in range(folds):
        tvt = [[], [], []] # train, val, test

        for sub in range(i, i + folds):
            target = 0 if sub < i + 3 else 1 if sub == i + 3 else 2
            sub = sub % folds
            p_size, n_size = len(p_fold[sub]), len(n_fold[sub])

            if target == 0:
                # If target is training set,
                # positive and negative data size should be the same
                if sampling == 'over-sampling':
                    if p_size - n_size > 0:
                        new_fold = copy.deepcopy(n_fold[sub])
                        for i3 in range(n_size, p_size):
                            new_fold.append(new_fold[i3 % n_size])

                        tvt[target] += new_fold
                        tvt[target] += p_fold[sub]
                    elif p_size - n_size < 0:
                        new_fold = copy.deepcopy(p_fold[sub])
                        for i3 in range(p_size, n_size):
                            new_fold.append(new_fold[i3 % p_size])

                        tvt[target] += new_fold
                        tvt[target] += n_fold[sub]
                    else:
                        tvt[target] += p_fold[sub]
                        tvt[target] += n_fold[sub]
                elif sampling == 'under-sampling':
                    tvt[target] += under_p_fold[sub]
                    tvt[target] += under_n_fold[sub]
                elif sampling == 'no-sampling':
                    tvt[target] += p_fold[sub]
                    tvt[target] += n_fold[sub]
            else:
                tvt[target] += p_fold[sub] + n_fold[sub]

        distributions[i] = tvt

    return distributions

def set_idx_to_name(id):
    if id == 0:
        return 'train'
    elif id == 1:
        return 'val'
    elif id == 2:
        return 'test'
    else:
        raise

def day_idx_to_name(id):
    if id == 0:
        return '42d'
    elif id == 1:
        return '90d'
    elif id == 2:
        return '365d'
    else:
        raise


def extract_ids_only(distributions):
    dis = list(distributions.values())
    simple_list = []

    for day_i, folds in enumerate(dis):
        days = day_idx_to_name(day_i)
        # 5 folds
        for fold_i, sets in enumerate(folds):
            # train, val, test
            for set_i, set_ in enumerate(sets):
                name = set_idx_to_name(set_i)
                set_ = np.array(set_)[:, 0]
                np.random.shuffle(set_)
                print('{}_{}_{} length is {}'.format(days, name, fold_i + 1, len(set_)))
                dis[day_i][fold_i][set_i] = set_
                set_ = ['{}_{}_{}'.format(days, name, fold_i + 1)] + set_.tolist()
                simple_list.append(set_)

    return simple_list


def days_distributer(data_path, folds=5, sampling='over-sampling'):
    distributions = {
        "42d": [],
        "90d": [],
        "365d": []
    }

    res = pd.read_csv(data_path)
    data = res.values
    data = np.array(data, dtype=np.int32)
    # header = res.columns.to_numpy()

    x_data, y_42d, y_90d, y_365d = data[:, :-3], data[:, -3], data[:, -2], data[:, -1]

    print('42d')
    distributions['42d'] = folds_distributer(y_42d, folds, sampling)
    distribution_test(distributions['42d'], sampling)

    print('90d')
    distributions['90d'] = folds_distributer(y_90d, folds, sampling)
    distribution_test(distributions['90d'], sampling)

    print('365d')
    distributions['365d'] = folds_distributer(y_365d, folds, sampling)
    distribution_test(distributions['365d'], sampling)

    return distributions

def csv_tester(path, origin):
    test = pd.read_csv(path, header=None)
    data = test.values.tolist()
    test_data = []
    new_data = []
    for each in data:
        each = [i for i in each if str(i) != 'nan']
        test_data.append(each)
        each = each[1:]
        new_data.append(each)
    assert str(origin) == str(origin)
    return new_data

if __name__ == "__main__":
    folds = 5
    # sampling = 'over-sampling'
    # sampling = 'under-sampling'
    sampling = 'no-sampling'
    clinical_data_path = "/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/RT_spine_NESMS_info/all.csv"
    distributions = days_distributer(clinical_data_path, folds, sampling)
    simple_distributions = extract_ids_only(distributions)

    with open('./datasets/train_val_test_split_{}.csv'.format(sampling), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(simple_distributions)
        fp.close()
    csv_tester('./datasets/train_val_test_split.csv', simple_distributions)