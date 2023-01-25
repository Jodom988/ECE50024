import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from share import *

def parta():
    xs = np.linspace(-3, 3, 100)
    vals = normal_list(xs)
    
    plt.plot(xs, vals)
    # plt.show()
    plt.savefig("figures/1-a.png")

def partb():
    
    rand_vals = np.array( [np.random.normal(0, 1) for i in range(1000)] )
    
    mu, std_dev = sp.stats.norm.fit(rand_vals)
    print(mu, std_dev)

    xs = np.array(np.linspace(-3, 3, 100))
    pdf = sp.stats.norm.pdf(xs, loc=0, scale=1)
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.hist(rand_vals, bins=4, density=True)
    ax1.plot(xs, pdf)

    ax2 = plt.subplot(2, 1, 2)
    ax2.hist(rand_vals, bins=1000, density=True)
    ax2.plot(xs, pdf)

    # plt.show()
    plt.savefig("figures/1-b.png")

def partc():

    rand_vals = np.array( [np.random.normal(0, 1) for i in range(1000)] )
    j(rand_vals, 5)

    bin_counts = np.array(range(1, 201))
    js = np.array([])
    for bin_count in bin_counts:
        js = np.append(js, j(rand_vals, bin_count))

    best_j = min(js)
    best_bin_count = bin_counts[np.where(js == best_j)[0][0]]

    print("Number of bins: %d " % best_bin_count)

    plt.plot(bin_counts, js)
    plt.plot([best_bin_count], [best_j], 'or')
    plt.savefig("figures/1-c-i.png")
    plt.show()

    mu, std_dev = sp.stats.norm.fit(rand_vals)
    print(mu, std_dev)

    xs = np.array(np.linspace(-3, 3, 100))
    pdf = sp.stats.norm.pdf(xs, loc=0, scale=1)
    
    ax1 = plt.subplot(1, 1, 1)
    ax1.hist(rand_vals, bins=best_bin_count, density=True)
    ax1.plot(xs, pdf)

    plt.savefig("figures/1-c-ii.png")
    plt.show()

def j(data, num_bins):
    '''
        num_bins = m
    '''
    max_val = np.max(data)
    min_val = np.min(data)
    n = len(data)
    bins = np.linspace(min_val, max_val, num_bins+1)
    bins[0] -= np.abs(bins[0]) * .001

    h = (max_val - min_val) / num_bins
    t1 = 2 / (h * (n - 1))
    t2 = (n + 1) / (h * (n - 1))

    t2_sum = 0
    tot_count = 0
    tot_prob = 0
    for i in range(0, num_bins):
        count_in_range = 0
        for sample in data:
            if sample > bins[i] and sample <= bins[i+1]:
                count_in_range += 1
        prob = count_in_range / n
        tot_prob += prob
        tot_count += count_in_range
        t2_sum += prob ** 2

    # print("Total count: %d, Tot prob: %f" % (tot_count, tot_prob))

    return t1 - (t2 * t2_sum)

def main():
    partc()

if __name__ == '__main__':
    main()
