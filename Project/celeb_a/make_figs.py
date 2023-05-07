
import numpy as np
import pandas as df
import matplotlib.pyplot as plt

from cond_gan_celeba import *


def vary_labels():

    gan = GAN.load('trials/gan_labels_3_stopped_2')

    count = 5

    scale_asscend = np.linspace(0, 1, count)
    scale_descend = scale_asscend[::-1]

    np.random.seed(7)
    latent = np.random.normal(0, 0.5, 96).reshape((1, 96))

    fig, axarr = plt.subplots(count, count, figsize=(8, 8))
    for row in range(count):
        for col in range(count):
            ax = axarr[row, col]
            male = scale_descend[row]
            female = scale_asscend[row]
            not_smiling = scale_descend[col]
            smiling = scale_asscend[col]

            label_arr = np.array([male, female, smiling, not_smiling]).reshape((1, 4))

            img = gan.generator.predict((latent, label_arr), verbose=3)[0]

            # ax.set_xlabel('%.1f,%.1f,%.1f,%.1f' % (male, female, smiling, not_smiling))
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.subplots_adjust(left=0, bottom=.1, right=1, top=1, wspace=.1, hspace=.25)
    plt.savefig('figures/label_variation.png')
    plt.show()



def example_generations():
    gan = GAN.load('trials/gan_labels_3_stopped_2')

    rows = 10

    labels = list()
    labels.append(np.array([1, 0, 1, 0]).reshape((1, 4)))
    labels.append(np.array([1, 0, 0, 1]).reshape((1, 4)))
    labels.append(np.array([0, 1, 1, 0]).reshape((1, 4)))
    labels.append(np.array([0, 1, 0, 1]).reshape((1, 4)))
    
    fig, axarr = plt.subplots(rows, 4, figsize=(4, 8))
    np.random.seed(10)
    for row in range(rows):
        for col in range(4):
            latent = np.random.normal(0, 0.5, 96).reshape((1, 96))
            img = gan.generator.predict((latent, labels[col]), verbose=3)[0]
            ax = axarr[row, col]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                if col == 0:
                    ax.set_title('Male\nSmiling')
                elif col == 1:
                    ax.set_title('Male\nNot Smiling')
                elif col == 2:
                    ax.set_title('Female\nSmiling')
                elif col == 3:
                    ax.set_title('Female\nNot Smiling')

    plt.subplots_adjust(left=0, bottom=0, right=1, top=.92, wspace=0, hspace=.17)
    plt.savefig('figures/example_generations.png')
    plt.show()


def smooth_data(data):
    WINDOW_SIZE = 50

    smoothed = list()
    for i in range(len(data)):
        if i < WINDOW_SIZE:
            smoothed.append(np.mean(data[:i+1]))
        else:
            smoothed.append(np.mean(data[i-WINDOW_SIZE:i+1]))

    return smoothed

def plot_training_progress_celeba():
    fname = 'trials/gan_labels_3_stopped_2/progress_post_discriminator.csv'
    df = pd.read_csv(fname)

    df['mae_real'] = 1 - df['avg_confidence_real']
    df['mae_fake'] = 1 - df['avg_confidence_fake']

    fig, axarr = plt.subplots(4, 1, figsize=(6, 8))
    ax = axarr[0]
    ax.plot(df['epoch_pct'], df['mae_real'], c='g')
    ax.plot(df['epoch_pct'], df['mae_fake'], c='r')
    ax.grid()
    ax.set_ylim(0, .6)
    ax.set_title('Discriminator MAE')
    ax.set_ylabel('MAE')
    ax.set_xlabel('Epoch')
    ax.legend(['Real Samples', 'Generated Samples'])

    ax = axarr[1]
    ax.plot(df['epoch_pct'], smooth_data(df['mae_real']), c='g')
    ax.plot(df['epoch_pct'], smooth_data(df['mae_fake']), c='r')
    ax.grid()
    ax.set_ylim(0, .6)
    ax.set_title('Discriminator MAE (Smoothed)')
    ax.set_ylabel('MAE')
    ax.set_xlabel('Epoch')

    ax = axarr[2]
    ax.plot(df['epoch_pct'], df['loss_real'], c='g')
    ax.plot(df['epoch_pct'], df['loss_fake'], c='r')
    ax.grid()
    ax.set_title('Discriminator Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')

    ax = axarr[3]
    ax.plot(df['epoch_pct'], smooth_data(df['loss_real']), c='g')
    ax.plot(df['epoch_pct'], smooth_data(df['loss_fake']), c='r')
    ax.grid()
    ax.set_title('Discriminator Loss (Smoothed)')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')

    plt.subplots_adjust(left=0.1, bottom=0.057, right=0.9, top=0.955, hspace=0.736)
    plt.savefig('figures/training_progress_mnist.png')
    plt.show()
    

def plot_training_progress_mnist():
    fname = '../mnist_testing/progress.csv'
    df = pd.read_csv(fname)

    fig, axarr = plt.subplots(4, 1, figsize=(6, 8))
    ax = axarr[0]
    ax.plot(df['epoch_pct'], df['mae'])
    ax.grid()
    ax.set_ylim(0, .6)
    ax.set_title('Discriminator MAE')
    ax.set_ylabel('MAE')
    ax.set_xlabel('Epoch')

    ax = axarr[1]
    ax.plot(df['epoch_pct'], smooth_data(df['mae']))
    ax.grid()
    ax.set_ylim(0, .6)
    ax.set_title('Discriminator MAE (Smoothed)')
    ax.set_ylabel('MAE')
    ax.set_xlabel('Epoch')

    ax = axarr[2]
    ax.plot(df['epoch_pct'], df['loss'])
    ax.grid()
    ax.set_title('Discriminator Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')

    ax = axarr[3]
    ax.plot(df['epoch_pct'], smooth_data(df['loss']))
    ax.grid()
    ax.set_title('Discriminator Loss (Smoothed)')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')

    plt.subplots_adjust(left=0.1, bottom=0.057, right=0.9, top=0.955, hspace=0.736)
    plt.savefig('figures/training_progress_mnist.png')
    plt.show()

def main():
    # vary_labels()
    # example_generations()
    # plot_training_progress_celeba()
    plot_training_progress_mnist()

if __name__ == '__main__':
    main()