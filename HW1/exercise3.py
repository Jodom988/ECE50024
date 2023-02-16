import numpy as np
from scipy.special import eval_legendre

from scipy.optimize import linprog

import matplotlib.pyplot as plt

def part_a_to_c():
    num_data = 50
    x = np.linspace(-1,1,num_data) # 50 points in the interval [-1,1]
    c = np.array([-0.001, 0.01, 0.55, 1.5, 1.2])
    y = (c[4]*eval_legendre(4, x)) + (c[3]*eval_legendre(3, x)) + (c[2]*eval_legendre(2, x)) + (c[1]*eval_legendre(1, x)) + (c[0]*eval_legendre(0, x))
    y = y + [np.random.normal(0, 0.2 ** 2) for _ in range(num_data)]

    X = np.zeros((num_data, 5))
    for row in range(num_data):
        for col in range(5):
            X[row, col] = eval_legendre(col, x[row])

    paren = np.matmul(np.matrix.transpose(X), X)
    weights = np.matmul(np.matmul(np.linalg.inv(paren), np.matrix.transpose(X)), y)

    print(c)
    print(weights)

    y_predicted = (weights[4]*eval_legendre(4, x)) + (weights[3]*eval_legendre(3, x)) + (weights[2]*eval_legendre(2, x)) + (weights[1]*eval_legendre(1, x)) + (weights[0]*eval_legendre(0, x))


    plt.scatter(x, y, marker='.')
    plt.plot(x, y_predicted, color='orange')
    plt.savefig("figures/3-c.png")
    plt.show()

def part_d():
    num_data = 50
    x = np.linspace(-1,1,num_data) # 50 points in the interval [-1,1]
    c = np.array([-0.001, 0.01, 0.55, 1.5, 1.2])
    y = (c[4]*eval_legendre(4, x)) + (c[3]*eval_legendre(3, x)) + (c[2]*eval_legendre(2, x)) + (c[1]*eval_legendre(1, x)) + (c[0]*eval_legendre(0, x))
    y = y + [np.random.normal(0, 0.2 ** 2) for _ in range(num_data)]

    idx = [10,16,23,37,45] # these are the locations of the outliers
    y[idx] = 5 # set the outliers to have a value 5

    X = np.zeros((num_data, 5))
    for row in range(num_data):
        for col in range(5):
            X[row, col] = eval_legendre(col, x[row])

    paren = np.matmul(np.matrix.transpose(X), X)
    weights = np.matmul(np.matmul(np.linalg.inv(paren), np.matrix.transpose(X)), y)

    print(c)
    print(weights)

    y_predicted = (weights[4]*eval_legendre(4, x)) + (weights[3]*eval_legendre(3, x)) + (weights[2]*eval_legendre(2, x)) + (weights[1]*eval_legendre(1, x)) + (weights[0]*eval_legendre(0, x))


    plt.scatter(x, y, marker='.')
    plt.plot(x, y_predicted, color='orange')
    plt.savefig("figures/3-d.png")
    plt.show()

def part_f():
    num_data = 50
    x = np.linspace(-1,1,num_data) # 50 points in the interval [-1,1]
    c = np.array([-0.001, 0.01, 0.55, 1.5, 1.2])
    y = (c[4]*eval_legendre(4, x)) + (c[3]*eval_legendre(3, x)) + (c[2]*eval_legendre(2, x)) + (c[1]*eval_legendre(1, x)) + (c[0]*eval_legendre(0, x))
    y = y + [np.random.normal(0, 0.2 ** 2) for _ in range(num_data)]

    idx = [10,16,23,37,45] # these are the locations of the outliers
    y[idx] = 5 # set the outliers to have a value 5

    idx = [5, 6]
    y[idx] = 3

    c = np.zeros(5 + num_data)
    for i in range(5, 5+num_data):
        c[i] = 1

    # print((c))
    
    A = np.zeros((num_data*2, 5+num_data))
    for row in range(num_data):
        for col in range(5):
            A[row, col] = eval_legendre(col, x[row])
    
    for row in range(num_data, num_data*2):
        for col in range(5):
            A[row, col] = -eval_legendre(col, x[row-num_data])

    for i in range(num_data):
        row = i
        col = 5 + i
        A[row, col] = -1
        A[row+num_data, col] = -1

    # for row in range(num_data*2):
    #     for col in range(num_data+5):
    #         print("%8.2f" % A[row, col], end=", ")
    #     print("")

    b = np.zeros(num_data * 2)
    for i in range(num_data):
        b[i] = y[i]
        b[i+num_data] = -y[i]

    # print(b)

    weights = linprog(c, A_ub=A, b_ub=b).x

    y_predicted = (weights[4]*eval_legendre(4, x)) + (weights[3]*eval_legendre(3, x)) + (weights[2]*eval_legendre(2, x)) + (weights[1]*eval_legendre(1, x)) + (weights[0]*eval_legendre(0, x))


    plt.scatter(x, y, marker='.')
    plt.plot(x, y_predicted, color='orange')
    plt.savefig("figures/3-f-quiz.png")
    plt.show()

def main():
    part_f()

if __name__ == '__main__':
    main()
