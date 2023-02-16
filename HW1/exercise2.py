import matplotlib.pyplot as plt
import numpy as np

def g(x1, x2):
    term1 = (x1-2) * (2*x1 - x2 + 2)
    term2 = (x2-6) * (-x1 + 2*x2 - 10)

    return (term1 + term2) / 6

def g_quiz(x1, x2):
    term1 = 2 * x1 * x1
    term2 = 2*x1 * x2
    term3 = 2 * x2 * x2
    return (term1 - term2 + term3) / 3

def f(x1, x2):
    coef = 1 / np.sqrt((2 * np.pi) ** 2 * np.sqrt(10))
    return coef * np.exp(-g_quiz(x1, x2))

def f_temp(x1, x2):
    return x1 + x2

def parta():
    density = 1000
    x1s = np.linspace(-5, 5, density)
    x2s = np.linspace(-5, 5, density)
    X1s, X2s = np.meshgrid(x1s, x2s)

    ys = []
    for i in range(X1s.shape[0]):
        current_y = []
        for j in range(X2s.shape[1]):
            current_y.append(f(X1s[i][j], X2s[i][j]))
        ys.append(current_y)
        # print(current_y)
        # print(ys)

        # print("====================")

    ys = np.array(ys)
    # print(ys)
    fig, ax = plt.subplots()
    cs = ax.contour(X1s, X2s, ys, 6)
    ax.clabel(cs, inline=True, fontsize=10)
    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 5))
    plt.savefig('figures/2-1-ii-quiz.png')
    plt.show()

def partc():
    # part i
    samples = np.random.multivariate_normal(np.zeros(2), np.identity(2), 5000)
    plt.scatter(samples[:, 0], samples[:, 1], marker='.')
    plt.savefig('figures/2-c-i.png')
    plt.show()

    samples_transformed = np.random.multivariate_normal([2, 6], [[2, 1], [1, 2]], 5000)

    plt.scatter(samples_transformed[:, 0], samples_transformed[:, 1], marker='.')
    plt.xlim(-1, 6)
    plt.ylim(0, 10)
    plt.savefig('figures/2-c-ii.png')
    plt.show()

    pass
    

def main():
    parta()
    # partc()

if __name__ == '__main__':
    main()