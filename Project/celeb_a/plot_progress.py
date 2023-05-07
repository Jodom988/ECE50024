import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WINDOW_SIZE = 50
# UPDATES_PER_EPOCH = 4670
UPDATES_PER_EPOCH = 1176

def smooth_data(data):
    smoothed = list()
    for i in range(len(data)):
        if i < WINDOW_SIZE:
            smoothed.append(np.mean(data[:i+1]))
        else:
            smoothed.append(np.mean(data[i-WINDOW_SIZE:i+1]))

    return smoothed

def main():
    fname = sys.argv[1]
    df = pd.read_csv(fname)
    df['acc_tot'] = (df['acc_real'] + df['acc_fake']) / 2
    df['updates'] = df['epoch_pct'] * UPDATES_PER_EPOCH

    x_axis_col = 'epoch_pct'

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    plt.plot([min(df[x_axis_col]), max(df[x_axis_col])], [0.5, 0.5], c='gray', linestyle='--')
    # plt.plot(df[x_axis_col], df['acc_real'], c='g', linestyle=':')
    # plt.plot(df[x_axis_col], df['acc_fake'], c='r', linestyle=':')
    plt.plot(df[x_axis_col], df['avg_confidence_real'], c='g', linestyle='-')
    plt.plot(df[x_axis_col], df['avg_confidence_fake'], c='r', linestyle='-')
    plt.grid(visible=True)
    plt.ylim([-.1, 1.1])
    
    plt.subplot(2, 2, 2)
    plt.plot([min(df[x_axis_col]), max(df[x_axis_col])], [0.5, 0.5], c='gray', linestyle='--')
    plt.plot(df[x_axis_col], smooth_data(df['acc_real']), c='g', linestyle=':')
    plt.plot(df[x_axis_col], smooth_data(df['acc_fake']), c='r', linestyle=':')
    plt.plot(df[x_axis_col], smooth_data(df['avg_confidence_real']), c='g', linestyle='-')
    plt.plot(df[x_axis_col], smooth_data(df['avg_confidence_fake']), c='r', linestyle='-')
    plt.legend(['', 'accuracy real', 'accuracy fake', 'confidence real', 'confidence fake'])
    plt.ylim([-.1, 1.1])
    plt.grid(visible=True)

    plt.subplot(2, 2, 3)
    plt.plot(df[x_axis_col], df['loss_real'], c='g')
    plt.plot(df[x_axis_col], df['loss_fake'], c='r')
    # plt.semilogy()
    plt.grid(visible=True)

    plt.subplot(2, 2, 4)
    plt.plot(df[x_axis_col], smooth_data(df['loss_real']), c='g')
    plt.plot(df[x_axis_col], smooth_data(df['loss_fake']), c='r')
    # plt.semilogy()
    plt.grid(visible=True)

    plt.savefig('progress.png')
    # plt.show()

    pass

if __name__ == '__main__':
    main()