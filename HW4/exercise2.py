import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cvxpy as cvx

LAMBDA = 0.0001

def get_data(fname):
    with open(fname, "r") as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    data = [x.split("   ") for x in data]
    x1 = np.array([float(x[0]) for x in data]).reshape((len(data), 1))
    x2 = np.array([float(x[1]) for x in data]).reshape((len(data), 1))
    
    return np.hstack((x1, x2))

def exb_to_c():

    # Get the data
    class0 = get_data("homework4_class0.txt")
    class1 = get_data("homework4_class1.txt")
    y0 = np.zeros((class0.shape[0], 1))
    y1 = np.ones((class1.shape[0], 1))

    # Format the data
    x = np.vstack((class0, class1))
    y = np.vstack((y0, y1)).reshape((y0.shape[0]+ y1.shape[0], 1))

    bias = np.ones(x.shape[0]).T.reshape((x.shape[0], 1))   # Add bias
    X = np.hstack((x, bias))

    # Solve problem with cvxpy
    theta = cvx.Variable((X.shape[1], 1))
    t1 = cvx.sum(cvx.multiply(y, X @ theta))
    t2 = cvx.sum(cvx.log_sum_exp( cvx.hstack([np.zeros((X.shape[0],1)), X @ theta]), axis=1 ))
    t3 = (LAMBDA * cvx.sum_squares(theta))
    loss = - (1 / X.shape[0]) * (t1 - t2) + t3
    
    prob = cvx.Problem(cvx.Minimize(loss))
    prob.solve()
    
    w = theta.value
    print(w)
    xs = np.linspace(-5, 10, 100)
    ys = - (w[0] * xs + w[2]) / w[1]

    plt.scatter(class0[:, 0], class0[:, 1], color="red")
    plt.scatter(class1[:, 0], class1[:, 1], color="blue")
    plt.plot(xs, ys, color="black")
    plt.savefig("2-c.png")
    plt.show()

def get_mu_sigma(data):
    mu = np.mean(data, axis=0)
    sigma = np.cov(data.T)

    return mu, sigma

def get_likelihood(mu, sigma, sigma_inv, log_sigma, log_pi, x):
    x_minus_mu = x - mu
    x_minus_mu_t = np.transpose(x_minus_mu)

    term1 = -0.5 * np.matmul(np.matmul(x_minus_mu_t, sigma_inv), x_minus_mu)[0, 0]
    term2 = log_pi
    term3 = log_sigma

    return term1 + term2 + term3

def exd():
    class0 = get_data("homework4_class0.txt")
    class1 = get_data("homework4_class1.txt")

    mu0, sigma0 = get_mu_sigma(class0)
    mu1, sigma1 = get_mu_sigma(class1)

    pi_0 = len(class0[0]) / (len(class0[0]) + len(class1[0]))
    pi_1 = len(class1[0]) / (len(class0[0]) + len(class1[0]))

    print(mu0)
    print(mu1)

    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    log_sigma0 = -0.5 * np.log(np.linalg.det(sigma0))
    log_sigma1 = -0.5 * np.log(np.linalg.det(sigma1))
    log_pi_0 = np.log(pi_0)
    log_pi_1 = np.log(pi_1)

    grid_xs = np.linspace(-5, 10, 100)
    grid_ys = np.linspace(-5, 10, 100)
    grid_zs = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            x = np.array([[grid_xs[i]], [grid_ys[j]]])
            likelihood0 = get_likelihood(mu0, sigma0, sigma0_inv, log_sigma0, log_pi_0, x)
            likelihood1 = get_likelihood(mu1, sigma1, sigma1_inv, log_sigma1, log_pi_1, x)
            grid_zs[j, i] = likelihood1 / likelihood0

    plt.scatter(class0[:, 0], class0[:, 1], color="red")
    plt.scatter(class1[:, 0], class1[:, 1], color="blue")
    plt.contour(grid_xs, grid_ys, grid_zs, levels=[1])
    plt.title("Bayesian Decision Rule")
    plt.savefig("2-d.png")
    plt.show()

def main():
    # exb_to_c()
    exd()

if __name__ == "__main__":
    main()