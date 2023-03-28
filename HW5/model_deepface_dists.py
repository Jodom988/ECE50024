import argparse
import random 
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import cvxpy as cvx

def get_data_by_img_trial(fname):
    with open(fname, "r") as f:
        lines = f.readlines()

    data = dict()
    for i in (range(len(lines))):
        line = lines[i]
        line = line.split(",")

        img_id = int(line[0])
        if img_id not in data:
            data[img_id] = (list(), list())

        data_class = int(line[1])

        dists = line[2:]
        dists = [float(d) for d in dists]
        dists = np.array(dists)

        if data_class == 0:
            data[img_id][0].append(dists)
        elif data_class == 1:
            data[img_id][1].append(dists)
        else:
            raise ValueError("Invalid class")
    
    return data

def get_data_pts_from_img_trials(data):
    all_trues = list()
    all_falses = list()
    for img_id, trial in data.items():
        class_false, class_true = get_data_pts_from_img_trial(trial)

        [all_trues.append(pt) for pt in class_true]
        [all_falses.append(pt) for pt in class_false]
    
    return all_trues, all_falses

def get_data_pts_from_img_trial(trial):
    class_false, class_true = trial
    return class_false, class_true

def feature_engineer(img_trials):
    
    data = list()
    tot_true_metrics = None
    tot_false_metrics = None
    for img_id, trial in img_trials.items():
        # if img_id == 59938:
        class_true, class_false = trial
        true_metrics, false_metrics = feature_engineer_trial(class_true, class_false)

        if tot_true_metrics is None:
            tot_true_metrics = true_metrics
            tot_false_metrics = false_metrics
        else:
            tot_true_metrics = np.vstack([tot_true_metrics, true_metrics])
            tot_false_metrics = np.vstack([tot_false_metrics, false_metrics])

    return tot_true_metrics, tot_false_metrics

def z_score(arr, mean, std):
    return (arr - mean) / std

def feature_engineer_trial(class_true, class_false):

    all_dists = list()
    for dist_list in class_true:
        for dist in dist_list:
            all_dists.append(dist)
    for dist_list in class_false:
        for dist in dist_list:
            all_dists.append(dist)

    mu = np.mean(all_dists)
    std = np.std(all_dists)

    for i, dist_list in enumerate(class_true):
        class_true[i] = z_score(dist_list, mu, std)
    for i, dist_list in enumerate(class_false):
        class_false[i] = z_score(dist_list, mu, std)

    true_metrics = feature_engineer_on_z_scores(class_true, len(all_dists))
    false_metrics = feature_engineer_on_z_scores(class_false, len(all_dists))

    return true_metrics, false_metrics

def feature_engineer_on_z_scores(data, total):
    data_metrics = list()
    n_features = 3
    for pt in data:
        best = np.max(pt)
        pct = pt.shape[0] / total
        med = np.median(pt)
        q1 = np.quantile(pt, 0.25)
        q3 = np.quantile(pt, 0.75)
        features = [best, med, pct]
        
        data_metrics.append([features])
    
    data_metrics = np.array(data_metrics).reshape(len(data), n_features)

    return data_metrics

def sigmoid_regression(class_true, class_false):
    
    y1 = np.ones((class_true.shape[0], 1))
    y0 = np.zeros((class_false.shape[0], 1))

    # Format the data
    x = np.vstack((class_true, class_false))
    y = np.vstack((y1, y0)).reshape((y1.shape[0]+ y0.shape[0], 1))

    bias = np.ones(x.shape[0]).T.reshape((x.shape[0], 1))   # Add bias
    X = np.hstack((x, bias))

    # Solve problem with cvxpy
    LAMBDA = 0.0001
    theta = cvx.Variable((X.shape[1], 1))
    t1 = cvx.sum(cvx.multiply(y, X @ theta))
    t2 = cvx.sum(cvx.log_sum_exp( cvx.hstack([np.zeros((X.shape[0],1)), X @ theta]), axis=1 ))
    t3 = (LAMBDA * cvx.sum_squares(theta))
    loss = - (1 / X.shape[0]) * (t1 - t2) + t3
    
    prob = cvx.Problem(cvx.Minimize(loss))
    prob.solve()
    
    coefs = theta.value

    return coefs

def split_test_train(data, test_size=0.2):
    data_l = list()
    for img_id, trials in data.items():
        data_l.append((img_id, trials))

    random.shuffle(data_l)
    n_test = int(len(data_l) * test_size)
    test = data_l[:n_test]
    train = data_l[n_test:]

    train_dict = dict()
    for img_id, trials in train:
        train_dict[img_id] = trials
    
    test_dict = dict()
    for img_id, trials in test:
        test_dict[img_id] = trials

    return train_dict, test_dict

def evaluate_model(img_trials, coefs):
    n_correct = 0
    n_total = 0
    for img_id, trial in img_trials.items():
        pred = predict_trial(trial, coefs)
        if pred == 0:
            n_correct += 1
        n_total += 1

    print("Accuracy: %.2f%%" % (n_correct / n_total * 100))

def predict_trial(img_trial, coefs):
    class_true, class_false = img_trial
    true_metrics, false_metrics = feature_engineer_trial(class_true, class_false)

    metrics = np.vstack([true_metrics, false_metrics])
    
    y = predict(metrics, coefs)
    print(np.argmin(y))
    return np.argmax(y)

def predict(xs, coefs):
    xs = map_to_1d(xs, coefs)
    y = 1 / (1 + np.exp(-xs))
    return xs, y

def map_to_1d(xs, coefs):
    if (xs.shape == (3,)):
        xs = xs.reshape((1, 3))
    bias = np.ones(xs.shape[0]).T.reshape((xs.shape[0], 1))
    xs = np.hstack([xs, bias])
    return xs @ coefs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_file", type=str)

    args = parser.parse_args()

    img_trials = get_data_by_img_trial(args.dist_file)
    img_trials_train, img_trials_test = split_test_train(img_trials)

    class_true, class_false = feature_engineer(img_trials_train)

    model_coefs = sigmoid_regression(class_true, class_false)

    evaluate_model(img_trials_test, model_coefs)        

    xs_true, ys_true = predict(class_true, model_coefs)
    xs_false, ys_false = predict(class_false, model_coefs)

    plt.scatter(xs_false, ys_false, label="False", color="red")
    plt.scatter(xs_true, ys_true, label="True", color="blue", marker='.')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()