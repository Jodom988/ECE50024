import matplotlib.pyplot as plt
import numpy as np
import sys

def smooth_data(times, temps, window_size=10):
    smoothed_times = list()
    smoothed_temps = list()

    first_time = times[0]
    last_time = times[-1]

    smoothed_times = list()
    curr_time = first_time
    while curr_time < last_time:
        smoothed_times.append(curr_time)
        curr_time += window_size

    for i, t in enumerate(smoothed_times):
        # find all the times within the window
        window_times = list()
        window_temps = list()
        window_start = t - window_size
        window_end = t + window_size
        for j, time in enumerate(times):
            if time >= window_start and time < window_end:
                window_times.append(time)
                window_temps.append(temps[j])

        smoothed_temps.append(np.mean(window_temps))

    return smoothed_times, smoothed_temps

def main():
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()

    train_acc = []
    for line in lines:
        train_acc.append(float(line) * 100)

    max_acc = max(train_acc)
    print("Max accuracy: %.2f" % (max_acc))

    epochs = np.arange(1, len(train_acc) + 1)
    smooth_window = len(train_acc) // 20 + 1
        
    smoothed_epochs, smoothed_train_acc = smooth_data(epochs, train_acc, window_size=smooth_window)

    ax = plt.subplot(2, 1, 1)
    ax.plot(epochs, train_acc)
    # ax.plot([epochs[0], epochs[-1]], [1, 1], 'r--')
    ax.set_ybound((0, None))
    ax.grid(True)
    ax.semilogx()
    ax = plt.subplot(2, 1, 2)
    ax.plot(smoothed_epochs, smoothed_train_acc)
    # ax.plot([epochs[0], epochs[-1]], [1, 1], 'r--')
    ax.set_ybound((0, None))
    ax.grid(True)
    ax.semilogx()
    # plt.show()
    plt.savefig("accuracy.png")
    

if __name__ == "__main__":
    main()