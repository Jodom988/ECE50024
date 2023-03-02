from exercise1 import load_train_data, normalize_data

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def part_a_to_b():
    data = load_train_data()
    normalize_data(data)

    X = np.array([data['stature_normalized'], data['bmi_normalized'], [1 for _ in range(data.shape[0])]])
    X = np.matrix.transpose(X)

    y = np.matrix.transpose(np.array(data['class']))

    paren = np.matmul(np.matrix.transpose(X), X)
    weights = np.matmul(np.matmul(np.linalg.inv(paren), np.matrix.transpose(X)), y)

    print(weights)

def part_c():
    data = load_train_data()
    normalize_data(data)

    A = np.array([data['stature_normalized'], data['bmi_normalized'], [1 for _ in range(data.shape[0])]])
    A = np.matrix.transpose(A)

    b = np.matrix.transpose(np.array(data['class']))

    x = cp.Variable(A.shape[1])
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [-15 <= x, x <= 10]
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    print(x.value)

    weights = x.value
    slope = weights[0] / -weights[1]
    intercept = weights[2] / -weights[1]
    xs = np.linspace(1.3, 2.0, 100)
    ys = slope * xs + intercept

    plt.scatter(data[data['gender'] == 'female']['stature_normalized'], data[data['gender'] == 'female']['bmi_normalized'], marker='x')
    plt.scatter(data[data['gender'] == 'male']['stature_normalized'], data[data['gender'] == 'male']['bmi_normalized'], marker='.')
    plt.plot(xs, ys, '--r')
    plt.xlabel("Stature (normalized)")
    plt.ylabel("BMI (normalized)")
    plt.legend(["Female", "Male"])
    plt.xlim(1.3, 2.0)
    plt.ylim(1, 9)

    plt.show()

def part_e_to_h():

    data = load_train_data()
    normalize_data(data)

    A = np.array([data['stature_normalized'], data['bmi_normalized'], [1 for _ in range(data.shape[0])]])
    A = np.matrix.transpose(A)

    y = np.matrix.transpose(np.array(data['class']))

    weights = np.zeros(A.shape[1])

    step_size = 0.00001
    beta = 0.9
    one_minus_beta = 1 - beta
    last_grad = None
    losses = []
    for i in range(50000):
        paren = np.matmul(A, weights) - y
        grad = 2 * np.matmul(np.matrix.transpose(A), paren)
        if last_grad is None:
            last_grad = grad

        # weights = weights - (step_size * grad)  # This line is for e and f
        weights = weights - (step_size * (beta * last_grad + one_minus_beta * grad))  # This line is for g and h
        last_grad = grad

        val = np.matmul(A, weights) - y
        loss = np.matmul(np.matrix.transpose(val), val)
        losses.append(loss)


    print(weights)

    # slope = weights[0] / -weights[1]
    # intercept = weights[2] / -weights[1]
    # xs = np.linspace(1.3, 2.0, 100)
    # ys = slope * xs + intercept

    # plt.scatter(data[data['gender'] == 'female']['stature_normalized'], data[data['gender'] == 'female']['bmi_normalized'], marker='x')
    # plt.scatter(data[data['gender'] == 'male']['stature_normalized'], data[data['gender'] == 'male']['bmi_normalized'], marker='.')
    # plt.plot(xs, ys, '--r')
    # plt.xlabel("Stature (normalized)")
    # plt.ylabel("BMI (normalized)")
    # plt.legend(["Female", "Male"])
    # plt.xlim(1.3, 2.0)
    # plt.ylim(1, 9)
    # plt.show()

    plt.plot(losses, linewidth=8)
    plt.semilogx()
    plt.show()
    
def main():
    # part_a_to_b()
    part_c()
    # part_e_to_h()
    pass

if __name__ == '__main__':
    main()