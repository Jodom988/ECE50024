from exercise1 import load_train_data, load_test_data, normalize_data

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm

import time
from threading import Thread

def two_norm(X):
    if X.ndim == 1:
        return sum([i ** 2 for i in X])
    else:
        return np.matmul(np.matrix.transpose(X), X)

def get_theta_lambda_async(lambd, outs):
    theta_lambda = get_theta_lambda(lambd)
    outs.append(lambd)
    outs.append(theta_lambda)

def get_theta_lambda(lambd):
    data = load_test_data()
    normalize_data(data)

    X = np.array([data['stature_normalized'], data['bmi_normalized'], [1 for _ in range(data.shape[0])]])
    X = np.matrix.transpose(X)
    y = np.matrix.transpose(np.array(data['class']))

    theta = cp.Variable(X.shape[1])
    objective = cp.Minimize( two_norm(X @ theta - y) + (lambd * two_norm(theta)) )
    constraints = []
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return theta.value

def part_a():
    data = load_test_data()
    normalize_data(data)

    X = np.array([data['stature_normalized'], data['bmi_normalized'], [1 for _ in range(data.shape[0])]])
    X = np.matrix.transpose(X)
    y = np.matrix.transpose(np.array(data['class']))

    results = dict()
    single_thread = True
    if single_thread:
        for lambd in tqdm(np.arange(0.1, 10, .1)):
            theta = get_theta_lambda(lambd)
            results[lambd] = theta
    else:
        threads = []
        for lambd in np.arange(0.1, 10, .1):
            out_list = []
            th = Thread(target=get_theta_lambda_async, args=(lambd, out_list))
            threads.append((th, out_list))
        
        for th, _ in threads:
            th.start()

        pbar = iter(tqdm(range(len(threads))))
        while len(threads) > 0:
            time.sleep(1)
            for thread in threads:
                worker, out_list = thread
                if not worker.is_alive():
                    worker.join()

                    results[out_list[0]] = out_list[1]

                    threads.remove(thread)
                        
                    try:
                        next(pbar)
                    except StopIteration:
                        pass
    
    fig, axarr = plt.subplots(3, 1)

    ax = axarr[0]
    xs = list()
    ys = list()
    for lambd, theta in results.items():
        xs.append(two_norm(theta))
        ys.append(two_norm(X @ theta - y))
    ax.plot(xs, ys)

    ax = axarr[1]
    xs = list()
    ys = list()
    for lambd, theta in results.items():
        xs.append(lambd)
        ys.append(two_norm(X @ theta - y))
    ax.plot(xs, ys)

    ax = axarr[2]
    for lambd, theta in results.items():
        xs.append(lambd)
        ys.append(two_norm(theta))
    ax.plot(xs, ys)
    
    plt.show()

    
def main():
    part_a()
    pass

if __name__ == '__main__':
    main()