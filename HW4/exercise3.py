from exercise2 import get_data

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cvxpy as cvx
from tqdm import tqdm

h = 1
LAMBDA = 0.0001

# x and y should be vectors of the same dimension
def kernel_func(x, y):
    x_minus_y = x - y
    x_minus_y_t = np.transpose(x_minus_y)
    return np.exp(- np.matmul(x_minus_y_t, x_minus_y) / h)

def rbf(x1, x2):  
  return np.exp(-np.sum((x1-x2)**2, axis=-1))

def get_k_grid(xs, ys):

    ks = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        x = np.array([x])
        for j, y in enumerate(ys):
            y = np.array([y])
            ks[i, j] = kernel_func(x, y)

    return ks

def ex_a():
    xs = list(range(47, 53))
    ys = list(range(47, 53))
    
    ks = get_k_grid(xs, ys)

    for i in range(len(xs)):
        for j in range(len(ys)):
            print("%6.2f" % ks[i, j], end=" ")
        print()

def ex_c():
    # Import the data
    class0 = get_data("homework4_class0.txt")
    class1 = get_data("homework4_class1.txt")
    y0 = np.zeros((class0.shape[0], 1))
    y1 = np.ones((class1.shape[0], 1))
    
    # Format the data
    data = np.vstack((class0, class1))
    print(data.shape)   # Prints (100, 2)
    y = np.vstack((y0, y1)).reshape((y0.shape[0]+ y1.shape[0], 1))
    print(y.shape)      # Prints (100, 1)

    # Create the kernel matrix
    K = np.zeros((data.shape[0], data.shape[0]))
    for r in range(data.shape[0]):
        row_data_pt = data[r]
        for c in range(data.shape[0]):
            col_data_pt = data[c]
            K[r, c] = kernel_func(row_data_pt, col_data_pt)

    # Format the problem
    alpha = cvx.Variable((data.shape[0], 1))
    t1 = y.T @ K @ alpha
    print("t1 shape: ", t1.shape)   # Prints (1, 1)

    ones = np.ones((1, data.shape[0]))
    inside_log = cvx.hstack([np.zeros((data.shape[0], 1)), K @ alpha])
    log_sum = cvx.log_sum_exp(inside_log, axis=1)
    print("log_sum shape: ", log_sum.shape)  # Prints (100,)

    t2 = ones @ log_sum
    print("t2 shape: ", t2.shape)  # Prints (1,)

    t3 = LAMBDA * cvx.quad_form(alpha, K)
    print("t3 shape: ", t3.shape)  # Prints (1, 1)

    loss = (-(1 / data.shape[0]) * (t1 - t2)) + t3
    print("loss shape: ", loss.shape)  # Prints (1, 1)

    # Solve the problem
    prob = cvx.Problem(cvx.Minimize(loss))
    prob.solve()

    alphas = alpha.value
    print("First 2 alphas: ")
    print("Alpha shape: ", alphas.shape)  # Prints (100, 1)
    print(alphas[0:2, 0]) # prints [-0.95245074 -1.21046707]
    
    # Get the surface of the kernel function
    res = 100
    grid_xs = np.linspace(-5, 10, res)
    grid_ys = np.linspace(-5, 10, res)
    grid_zs = np.zeros((res, res))

    vals = list()
    for i in tqdm(range(res)):
        for j in range(res):
            one_d_pt = 0
            for n in range(data.shape[0]):
                x_n = data[n]
                new_x = np.array([grid_xs[i], grid_ys[j]])
                one_d_pt += alphas[n, 0] * kernel_func(x_n, new_x)
            grid_zs[j, i] = 1 / (1 + np.exp(-one_d_pt)) # Switch j and i here because rows are xs and cols are ys

            vals.append(grid_zs[i, j])
    
    print("max: %.2f" % max(vals))
    print("min: %.2f" % min(vals))
    print("avg: %.2f" % np.mean(vals))
    print("std: %.2f" % np.std(vals))

    # Plot the data
    plt.scatter(class0[:, 0], class0[:, 1], color="red")
    plt.scatter(class1[:, 0], class1[:, 1], color="blue")
    plt.contour(grid_xs, grid_ys, grid_zs, levels=[.5], colors="black")
    plt.show()
    plt.savefig("3-d.png")


def main():
    # ex_a()
    ex_c()

if __name__ == "__main__":
    main()