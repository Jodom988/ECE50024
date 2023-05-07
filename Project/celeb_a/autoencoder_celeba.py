import argparse
import gc
import os
import time
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

IM_SHAPE = (32, 32, 3)
CHECKPOINT_DIR = 'trial_autoencoder'
PROGRESS_FNAME = 'progress.csv'
# LEARNING_RATE = 0.0005

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)

def load_data_pool(csv_fpath, img_dir, pct_train=0.8):

    with open(csv_fpath, 'r') as f:
        lines = f.readlines()

    # lines = lines[:10000]

    data = list()
    i = 0
    for line in tqdm(lines, desc="Loading data"):
        if i == 0:
            i += 1
            continue

        line = line.split(',')
        img_name = line[0]
        img_path = os.path.join(img_dir, img_name)

        data.append(img_path)
        i += 1

    np.random.seed(123)
    np.random.shuffle(data)

    idx = int(len(data) * pct_train)
    train_data = data[:idx]
    test_data = data[idx:]
    return train_data, test_data

def prepare_batch(data_pool, n):

    idxes = np.random.choice(list(range(len(data_pool))), n, replace=False)
    X_real_imgs = load_imgs_from_idxes(data_pool, idxes)

    X_imgs = X_real_imgs

    return X_imgs, X_imgs

def load_imgs_from_idxes(data_pool, idexs):
    batch_imgs = list()
    for idx in idexs:
        img_path = data_pool[idx]
        img = Image.open(img_path)
        # img = ImageOps.grayscale(img)
        img = img.resize((IM_SHAPE[0], IM_SHAPE[1]))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(IM_SHAPE)
        batch_imgs.append(img)

    batch_imgs = np.array(batch_imgs)
    return batch_imgs

def get_random_samples(n):
    samples = list()
    
    for _ in range(n):
        sample = np.random.rand(IM_SHAPE[0], IM_SHAPE[1], IM_SHAPE[2])
        samples.append(sample)

    samples = np.array(samples)
    return samples

def evaluate_model(model, X_test, epoch_pct, data_idx):
    Xs_test, ys_test = prepare_batch(X_test, 2048)
    loss, acc = model.evaluate(Xs_test, ys_test, verbose=3)

    with open(os.path.join(CHECKPOINT_DIR, PROGRESS_FNAME), 'a') as f:
        f.write("%f,%d,%f,%f\n" % (epoch_pct, data_idx, acc, loss))

def train_models(X_train, X_test, encoder, decoder, model, init_data_idx, batch_size = 2*8192):

    np.random.seed(1)
    view_imgs, _ = prepare_batch(X_test, 10)

    print("Starting training...")
    batch_idx = 0
    minibatch_size = 64
    data_idx = init_data_idx
    np.random.seed(int(time.time()))
    while True:
        # Train model
        Xs_train, ys_train = prepare_batch(X_train, batch_size)
        model.fit(Xs_train, ys_train, verbose=3, batch_size=minibatch_size)

        data_idx += batch_size

        # Clear up memory leaks
        gc.collect()
        tf.keras.backend.clear_session()


        if batch_idx % 3 == 0:
            # Evaluate models
            evaluate_model(model, X_test, data_idx / len(X_train), data_idx)

            # Polt some reconstructed samples
            reconstructed = model.predict(view_imgs, verbose=3)

            plt.figure(figsize=(20, 5))
            for i in range(10):
                ax = plt.subplot(2, 10, i+1)
                img = view_imgs[i]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])

                psnr = tf.image.psnr(view_imgs[i], reconstructed[i], max_val=1.0)
                ax = plt.subplot(2, 10, i+11)
                img = reconstructed[i]
                ax.imshow(img)
                ax.set_xlabel("PSNR: %.2f" % (psnr))
                ax.set_xticks([])
                ax.set_yticks([])

            plt.savefig(os.path.join(CHECKPOINT_DIR, "tmp.png"))
            plt.close()
            plt.clf()

            # print("Saving checkpoint models...")
            encoder.save(os.path.join(CHECKPOINT_DIR, "encoder.h5"))
            decoder.save(os.path.join(CHECKPOINT_DIR, "decoder.h5"))

        batch_idx += 1

def combine_models(generator, discriminator):
    input = layers.Input(IM_SHAPE)

    x = generator(input)
    outputs = discriminator(x)

    model = keras.Model(input, outputs)

    model.summary()

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def build_decoder():

    leakyrelu_alpha = 0.01

    inputs = layers.Input(shape=(4, 4, 64))

    d = layers.Conv2DTranspose(64, kernel_size=(2,2),  strides=(2,2))(inputs)       # 8,8
    # d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)

    d = layers.Conv2D(128, (2, 2), padding='same')(d)
    # d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)

    d = layers.Conv2DTranspose(128, kernel_size=(2,2),  strides=(2,2))(d)       # 16,16
    # d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)

    d = layers.Conv2D(128, (2, 2), padding='same')(d)
    # d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)

    d = layers.Conv2DTranspose(128, kernel_size=(2,2), strides=(2,2))(d)        # 32,32
    # d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)

    d = layers.Conv2D(128, (2, 2), padding='same')(d)
    # d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)
    
    img = layers.Conv2D(3, (2, 2), activation='sigmoid', padding='same')(d)                     # 32,32
    model = keras.Model(inputs, img, name='decoder')
    model.summary()

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def build_encoder():
    leaky_relualpha = 0.01

    inputs = layers.Input(shape=IM_SHAPE)

    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.LeakyReLU(leaky_relualpha)(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(leaky_relualpha)(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(leaky_relualpha)(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.LeakyReLU(leaky_relualpha)(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(leaky_relualpha)(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(leaky_relualpha)(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(leaky_relualpha)(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.LeakyReLU(leaky_relualpha)(x)


    model= keras.Model(inputs, x, name='encoder')

    model.summary()
    
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", '--load_checkpoints', action="store_true", dest='load_checkpoints')
    args = parser.parse_args()
    
    train_pool, test_pool = load_data_pool('../../data/Project/celeba_cropped_labels.csv', '../../data/Project/celebA_cropped/')

    if args.load_checkpoints:
        encoder = keras.models.load_model(os.path.join(CHECKPOINT_DIR, "encoder.h5"))
        decoder = keras.models.load_model(os.path.join(CHECKPOINT_DIR, "decoder.h5"))
        full_model = combine_models(encoder, decoder)

        df = pd.read_csv(os.path.join(CHECKPOINT_DIR, PROGRESS_FNAME))
        data_idx = df['data_idx'].iloc[-1]
    else:
        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
        os.makedirs(CHECKPOINT_DIR)

        encoder = build_encoder()
        decoder = build_decoder()
        full_model = combine_models(encoder, decoder)

        with open(os.path.join(CHECKPOINT_DIR, PROGRESS_FNAME), 'w') as f:
            f.write('epoch_pct,data_idx,mse,loss\n')
        data_idx = 0

    train_models(train_pool, test_pool, encoder, decoder, full_model, data_idx)

if __name__ == "__main__":
    main()