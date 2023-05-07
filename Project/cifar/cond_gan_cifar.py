import argparse
import gc
import os
import time
import shutil
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


IM_SHAPE = (32, 32, 3)
FILTER_LABELS = []
FILTER_LABELS_VALS = []
N_LABELS = 10
LATENT_DIM = 128
CHECKPOINT_DIR = 'gan_autoencoder_labels'
PROGRESS_POST_GEN_FNAME = 'progress_post_generator.csv'
PROGRESS_POST_DISC_FNAME = 'progress_post_discriminator.csv'
LEARNING_RATE = 0.00007
# LEARNING_RATE = 0.0005

LABEL_TO_NAME = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)

class GAN():

    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

        inputs_latent = layers.Input(shape=(LATENT_DIM,))
        inputs_label = layers.Input(shape=(N_LABELS,))

        x = self.generator([inputs_latent, inputs_label])
        outputs = self.discriminator([x, inputs_label])

        self.full_model = keras.Model([inputs_latent, inputs_label], outputs)
        self.full_model.summary()

        opt = keras.optimizers.Adam(LEARNING_RATE, 0.5)
        self.full_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'mean_absolute_error'])        

    def train_generator(self, Xs, ys, batch_size=32, verbose=3, shuffle=True):
        self.generator.trainable = True
        self.discriminator.trainable = False

        self.full_model.fit(Xs, ys, batch_size=batch_size, verbose=verbose, shuffle=shuffle)

    def train_discriminator(self, Xs, ys, batch_size=32, verbose=3, validation_data=None, shuffle=True):
        self.discriminator.trainable = True

        self.discriminator.fit(Xs, ys, batch_size=batch_size, verbose=verbose, validation_data=validation_data, shuffle=shuffle)

    def generate(self, latent, label, verbose=3):
        return self.generator.predict([latent, label], verbose=verbose)
    
    def save(self, dir):
        dir_backup = dir + '_backup'
        
        # copy dir to dir_backup
        if os.path.exists(dir_backup):
            shutil.rmtree(dir_backup)

        if os.path.exists(dir):
            shutil.copytree(dir, dir_backup)

        self.generator.save(os.path.join(dir, 'generator.h5'))
        self.discriminator.save(os.path.join(dir, 'discriminator.h5'))

    @staticmethod
    def load(gan_dir):
        try:
            generator = keras.models.load_model(os.path.join(gan_dir, 'generator.h5'))
            discriminator = keras.models.load_model(os.path.join(gan_dir, 'discriminator.h5'))

        except OSError:
            dir_backup = gan_dir + '_backup'
            generator = keras.models.load_model(os.path.join(dir_backup, 'generator.h5'))
            discriminator = keras.models.load_model(os.path.join(dir_backup, 'discriminator.h5'))


        return GAN(generator, discriminator)

def load_data_pool(fpaths):


    all_imgs = None
    all_labels = None

    for fpath in fpaths:
        with open(fpath, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        imgs = d[b'data']
        labels = np.array(d[b'labels']).reshape((-1, 1))

        labels_one_hot = np.zeros((labels.shape[0], N_LABELS))
        labels_one_hot[np.arange(labels.shape[0]), labels[:, 0]] = 1

        imgs = imgs.reshape((-1, 3, 32, 32))
        imgs = imgs.transpose((0, 2, 3, 1))

        imgs = imgs / 255.0

        if all_imgs is None:
            all_imgs = imgs
            all_labels = labels_one_hot
        else:
            all_imgs = np.vstack((all_imgs, imgs))
            all_labels = np.vstack((all_labels, labels_one_hot))

    return all_imgs, all_labels

def prepare_discriminator_batch(gan: GAN, data_pool, n, w_real=1.0, w_fake=1.0, w_swapped=1.0):
    n_real = int(n * (w_real / (w_real + w_fake + w_swapped)))
    n_swapped = int(n * (w_swapped / (w_real + w_fake + w_swapped)))
    n_fake = n - n_real - n_swapped

    if (n_real + n_swapped + n_fake) != n:
        raise ValueError("n_real + n_swapped + n_fake != n")

    X_imgs = None
    X_labels = None
    ys = None

    # get real samples
    if n_real != 0:
        idxes = np.random.choice(list(range(len(data_pool[0]))), n_real, replace=False)
        X_real_imgs, X_real_labels = load_imgs_from_idxes(data_pool, idxes)
        ys = np.ones((n_real, 1))

        X_imgs = X_real_imgs
        X_labels = X_real_labels

    # get generated samples
    if n_fake != 0:
        if gan is None:
            X_fake_imgs, X_fake_labels = get_random_samples(n_fake)
        else:
            latent_inputs, X_fake_labels = get_latent_space_inputs(n_fake)
            X_fake_imgs = gan.generate(latent_inputs, X_fake_labels)

        if X_imgs is None:
            X_imgs = X_fake_imgs
            X_labels = X_fake_labels
            ys = np.zeros((n_fake, 1))
        else:
            X_imgs = np.vstack((X_imgs, X_fake_imgs))
            X_labels = np.vstack((X_labels, X_fake_labels))
            ys = np.vstack((ys, np.zeros((n_fake, 1))))

    # get swapped samples
    if n_swapped != 0:
        idxes = np.random.choice(list(range(len(data_pool))), n_swapped, replace=False)
        X_swapped_imgs, X_swapped_labels = load_imgs_from_idxes(data_pool, idxes)
        
        np.random.shuffle(X_swapped_labels)
        X_swapped_labels = np.round(np.random.random(X_swapped_labels.shape))

        if X_imgs is None:
            X_imgs = X_swapped_imgs
            X_labels = X_swapped_labels
            ys = np.zeros((n_swapped, 1))
        else:
            X_imgs = np.vstack((X_imgs, X_swapped_imgs))
            X_labels = np.vstack((X_labels, X_swapped_labels))
            ys = np.vstack((ys, np.zeros((n_swapped, 1))))

        pass
    
    return (X_imgs, X_labels), ys

def load_imgs_from_idxes(data_pool, idexs):
    batch_imgs = list()
    batch_labels = list()
    for idx in idexs:
        img = data_pool[0][idx]
        batch_imgs.append(img)

        labels = data_pool[1][idx]
        batch_labels.append(labels)

    batch_imgs = np.array(batch_imgs)
    batch_labels = np.array(batch_labels)
    return batch_imgs, batch_labels

def get_random_samples(n):
    samples = list()
    labels = list()
    for _ in range(n):
        sample = np.random.rand(IM_SHAPE[0], IM_SHAPE[1], IM_SHAPE[2])
        samples.append(sample)

        label = np.random.rand(N_LABELS)
        labels.append(label)

    samples = np.array(samples)
    labels = np.array(labels)
    return samples, labels

def get_latent_space_inputs(n):
    inputs = list()
    labels = list()

    for i in range(n):
        input = np.random.normal(0, 0.5, LATENT_DIM)
        inputs.append(input)

        label = np.zeros(N_LABELS)
        label[np.random.randint(0, N_LABELS)] = 1

        labels.append(label)

    inputs = np.array(inputs)
    labels = np.array(labels)

    return inputs, labels

def pretrian_discriminator(gan: GAN, n, X_train, X_test):
    
    n_train = n
    n_test = n // 4

    Xs_train, ys_train = prepare_discriminator_batch(None, X_train, n_train, w_swapped=0)
    Xs_test, ys_test = prepare_discriminator_batch(None, X_test, n_test, w_swapped=0)

    print("Training initial discriminator...")
    gan.train_discriminator(Xs_train, ys_train, validation_data=(Xs_test, ys_test), verbose=1)
    print("Done!")

def evaluate_models(gan: GAN, X_test, epoch_pct, data_idx, fname):
    Xs_test, ys_test = prepare_discriminator_batch(gan, X_test, 2048, w_real=1.0, w_fake=0, w_swapped=0)
    loss_real, acc_real, mae_real = gan.discriminator.evaluate(Xs_test, ys_test, verbose=3)

    avg_confidence_real = np.mean(gan.discriminator.predict(Xs_test, verbose=3))

    # print()

    Xs_test, ys_test = prepare_discriminator_batch(gan, X_test, 2048, w_fake=1.0, w_real=0, w_swapped=0)
    loss_fake, acc_fake, mae_fake = gan.discriminator.evaluate(Xs_test, ys_test, verbose=3)
    avg_confidence_fake = 1 - np.mean(gan.discriminator.predict(Xs_test, verbose=3))

    with open(os.path.join(CHECKPOINT_DIR, fname), 'a') as f:
        f.write("%f,%d,%f,%f,%f,%f,%f,%f\n" % (epoch_pct, data_idx, acc_real, acc_fake, loss_real, loss_fake, avg_confidence_real, avg_confidence_fake))

def train_models(X_train, X_test, gan: GAN, init_data_idx, batch_size = 16):

    np.random.seed(2)
    view_real, _ = prepare_discriminator_batch(None, X_test, 10, w_real=1.0, w_fake=0, w_swapped=0)
    view_latent = get_latent_space_inputs(10)

    test_gen_Xs = get_latent_space_inputs(64)
    test_gen_ys = np.ones((64, 1))

    print("Starting training...")
    batch_idx = 0
    data_idx = init_data_idx
    np.random.seed(int(time.time()))
    while True:

        # Train generator
        mae = 1
        count = 0
        while mae > 0.1 and count < 64:
            Xs_train = get_latent_space_inputs(batch_size)
            ys_train = np.ones((batch_size, 1))

            gan.train_generator(Xs_train, ys_train, batch_size=batch_size, verbose=3)
            data_idx += batch_size / 2
            count += 1

            loss, acc, mae = gan.full_model.evaluate(test_gen_Xs, test_gen_ys, verbose=3)

        print("Generator:     %2d, %.2f, %.2f" % (count, acc, mae))
        

        # Evaluate post generator training
        if batch_idx % 10 == 0:
            evaluate_models(gan, X_test, data_idx / len(X_train), data_idx, PROGRESS_POST_GEN_FNAME)

        # Train discriminator
        mae = 1
        count = 0
        test_disc_Xs, test_disc_ys = prepare_discriminator_batch(gan, X_test, 64, w_swapped=0)
        while mae > 0.1 and count < 64:
            Xs_train, ys_train = prepare_discriminator_batch(gan, X_train, batch_size, w_swapped=0)

            gan.train_discriminator(Xs_train, ys_train, batch_size=batch_size, verbose=3)
            loss, acc, mae = gan.discriminator.evaluate(test_disc_Xs, test_disc_ys, verbose=3)
            data_idx += batch_size / 2
            count += 1

        print("Discriminator: %2d, %.2f, %.2f" % (count, acc, mae))


        # Clear up memory leaks
        gc.collect()
        tf.keras.backend.clear_session()

        # Evaluate post discriminator training
        if batch_idx % 10 == 0:
            evaluate_models(gan, X_test, data_idx / len(X_train), data_idx, PROGRESS_POST_DISC_FNAME)

        if batch_idx % 20 == 0:
            print("Updating view")
            # Polt some generated samples
            view_fake = gan.generator.predict(view_latent, verbose=3)
            sample_imgs = np.vstack((view_real[0], view_fake))
            sample_labels = np.vstack((view_real[1], view_latent[1]))
            y = np.vstack((np.ones((10, 1)), np.zeros((10, 1))))
            predictions = gan.discriminator.predict((sample_imgs, sample_labels), verbose=3)
            plt.figure(figsize=(20, 5))
            for i in range(10):
                ax = plt.subplot(2, 10, i+1)
                img = sample_imgs[i]
                ax.imshow(img)
                class_idx = np.argmax(view_real[1][i])
                class_name = LABEL_TO_NAME[class_idx]
                ax.set_xlabel("%.2f (%d)\n%s" % (predictions[i][0], y[i][0], class_name))
                ax.set_xticks([])
                ax.set_yticks([])
                
                ax = plt.subplot(2, 10, i+11)
                img = sample_imgs[i+10]
                ax.imshow(img)
                class_idx = np.argmax(view_latent[1][i])
                class_name = LABEL_TO_NAME[class_idx]
                ax.set_xlabel("%.2f (%d)\n%s" % (predictions[i+10][0], y[i+10][0], class_name))
                ax.set_xticks([])
                ax.set_yticks([])

            plt.savefig(os.path.join(CHECKPOINT_DIR, "tmp.png"))
            plt.close()
            plt.clf()
        
        if batch_idx % 20 == 0:
            print("Saving checkpoint models")
            gan.save(CHECKPOINT_DIR)
            print("Done")

        batch_idx += 1

def build_generator():

    leakyrelu_alpha = 0.01

    inputs_latent = layers.Input(shape=(LATENT_DIM,))
    inputs_labels = layers.Input(shape=(N_LABELS,))
    inputs = layers.concatenate([inputs_latent, inputs_labels])

    d = layers.Dense(128)(inputs)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)

    d = layers.Dense(256)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)

    d = layers.Dense(512)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)

    d = layers.Dense(1024)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)

    d = layers.Dense(2048)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)

    d = layers.Dense(4096)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)

    d = layers.Dense(128*8*8)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)
    d = layers.Reshape((8,8,128))(d)

    d = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)
    
    d = layers.Conv2D(128, (3, 3), padding='same')(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)

    d = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)
    
    d = layers.Conv2D(3, (7, 7), padding='same', activation='sigmoid')(d)
    
    model = keras.Model((inputs_latent, inputs_labels), d, name="latent_generator_model")
    model.summary()

    opt = keras.optimizers.Adam(LEARNING_RATE, 0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'mean_absolute_error'])

    return model

def build_discriminator():
    leakyrelu_alpha = 0.2


    inputs_img = layers.Input(shape=(IM_SHAPE))
    inputs_labels = layers.Input(shape=(N_LABELS,))

    x = layers.Conv2D(128, (3, 3), padding='same')(inputs_img)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)


    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, inputs_labels])

    x = layers.Dense(2048)(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    
    model= keras.Model((inputs_img, inputs_labels), out, name="latent_discriminator_model")

    model.summary()
    
    opt = keras.optimizers.Adam(LEARNING_RATE, 0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'mean_absolute_error'])
    
    return model




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", '--load_checkpoints', action="store_true", dest='load_checkpoints')
    args = parser.parse_args()

    train_pool = load_data_pool(['../../data/Project/cifar-10-batches-py/data_batch_%d' % i for i in range(1, 2)])
    test_pool = load_data_pool(['../../data/Project/cifar-10-batches-py/test_batch'])

    if args.load_checkpoints:
        gan = GAN.load(CHECKPOINT_DIR)

        df = pd.read_csv(os.path.join(CHECKPOINT_DIR, PROGRESS_POST_DISC_FNAME))
        data_idx = df['data_idx'].iloc[-1]
    else:
        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
        os.makedirs(CHECKPOINT_DIR)

        generator = build_generator()
        discriminator = build_discriminator()
        gan = GAN(generator, discriminator)

        with open(os.path.join(CHECKPOINT_DIR, PROGRESS_POST_DISC_FNAME), 'w') as f:
            f.write('epoch_pct,data_idx,acc_real,acc_fake,loss_real,loss_fake,avg_confidence_real,avg_confidence_fake\n')
        
        with open(os.path.join(CHECKPOINT_DIR, PROGRESS_POST_GEN_FNAME), 'w') as f:
            f.write('epoch_pct,data_idx,acc_real,acc_fake,loss_real,loss_fake,avg_confidence_real,avg_confidence_fake\n')
        data_idx = 0
        pretrian_discriminator(gan, 8192, train_pool, test_pool)


    train_models(train_pool, test_pool, gan, data_idx)

if __name__ == "__main__":
    main()