import matplotlib.pyplot as plt
import numpy as np
import sys

def smooth_data2(vals):
    vals = np.array(vals)
    num_points = 25
    xs = np.logspace(0, np.log10(len(vals)), num_points)

    return xs, np.interp(xs, np.arange(len(vals)), vals)

def main():

    n_data_sets = len(sys.argv) - 1

    datasets = []
    smoothed_data = []
    for i in range(n_data_sets):
        # Load in data
        with open(sys.argv[i + 1], 'r') as f:
            lines = f.readlines()

        train_acc = []
        for line in lines:
            train_acc.append(float(line) * 100)

        # Find the best accuracy
        max_acc = max(train_acc)
        print("Max accuracy: %.2f" % (max_acc))
    
        # Smooth the data
        smoothed_xs, smoothed_ys = smooth_data2(train_acc)

        datasets.append(train_acc)
        smoothed_data.append((smoothed_xs, smoothed_ys))

    # Plot the data
    ax = plt.subplot(2, 1, 1)
    for i in range(n_data_sets):
        ax.plot(datasets[i])
    ax.set_ybound((0, None))
    ax.grid(True)
    ax.semilogx()
    ax = plt.subplot(2, 1, 2)
    for i in range(n_data_sets):
        smoothed_xs, smoothed_ys = smoothed_data[i]
        ax.plot(smoothed_xs, smoothed_ys)
    ax.set_ybound((0, None))
    ax.grid(True)
    ax.semilogx()
    # plt.show()
    plt.savefig("accuracy.png")
    

if __name__ == "__main__":
    main()