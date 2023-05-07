import argparse
import gc
import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

IM_SHAPE = (32, 32, 3)
LATENT_DIM = 128
PROGRESS_FNAME = 'progress.csv'

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
        embeddings = np.array([float(x) for x in line[1:]])

        data.append((img_path, embeddings))
        i += 1

    np.random.seed(123)
    np.random.shuffle(data)

    idx = int(len(data) * pct_train)
    train_data = data[:idx]
    test_data = data[idx:]
    return train_data, test_data

def prepare_discriminator_batch(generator, data_pool, n, pct_real):
    n_real = int(n * pct_real)
    n_fake = n - n_real

    # first get real samples
    if n_real != 0:
        idxes = np.random.choice(list(range(len(data_pool))), n_real, replace=False)
        X_real_imgs, _ = load_imgs_from_idxes(data_pool, idxes)

    # then get generated samples
    if n_fake != 0:
        if generator is None:
            X_fake_imgs = get_random_samples(n_fake)
        else:
            X_fake_imgs = generator.predict(get_latent_space_inputs(n_fake), verbose=3)

    if n_real == 0:
        return X_fake_imgs, np.zeros((n_fake, 1))
    elif n_fake == 0:
        return X_real_imgs, np.ones((n_real, 1))
    else:
    
        Xs = np.vstack((X_real_imgs, X_fake_imgs))
        ys = np.vstack((np.ones((n_real, 1)), np.zeros((n_fake, 1))))

        return Xs, ys


def load_imgs_from_idxes(data_pool, idexs):
    batch_imgs = list()
    batch_embeddings = list()
    for idx in idexs:
        img_path = data_pool[idx][0]
        img = Image.open(img_path)
        # img = ImageOps.grayscale(img)
        img = img.resize((IM_SHAPE[0], IM_SHAPE[1]))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(IM_SHAPE)
        batch_imgs.append(img)

        embeddings = data_pool[idx][1]
        embeddings = np.array(embeddings)
        batch_embeddings.append(embeddings)

    batch_imgs = np.array(batch_imgs)
    batch_embeddings = np.array(batch_embeddings)
    return batch_imgs, batch_embeddings

def get_random_samples(n):
    samples = list()
    for _ in range(n):
        sample = np.random.rand(IM_SHAPE[0], IM_SHAPE[1], IM_SHAPE[2])
        samples.append(sample)

    samples = np.array(samples)
    return samples

def get_latent_space_inputs(n):
    inputs = list()
    for _ in range(n):
        input = np.random.normal(0, 0.5, LATENT_DIM)
        inputs.append(input)

    inputs = np.array(inputs)
    return inputs

def pretrian_discriminator(discriminator, n, X_train, X_test):
    
    n_train = n
    n_test = n // 4

    Xs_train, ys_train = prepare_discriminator_batch(None, X_train, n_train, 0.5)
    Xs_test, ys_test = prepare_discriminator_batch(None, X_test, n_test, 0.5)

    print("Training initial discriminator...")
    print(Xs_train.shape, ys_train.shape)
    print(Xs_test.shape, ys_test.shape)
    discriminator.fit(Xs_train, ys_train, epochs=2, validation_data=(Xs_test, ys_test), shuffle=True)
    print("Done!")

def randfloat(low, high):
    return np.random.rand() * (high - low) + low

def train_models(X_train, X_test, discriminator, generator, batch_size = 2048):

    # 1 = real, 0 = fake

    full_model = combine_models(generator, discriminator)

    view_real, _ = prepare_discriminator_batch(None, X_test, 10, 1)
    view_latent = get_latent_space_inputs(10)

    print("Starting training...")
    batch_idx = 0
    while True:

        # Train generator
        discriminator.trainable = False
        generator.trainable = True

        Xs_train = get_latent_space_inputs(batch_size)
        ys_train = np.ones((batch_size, 1))

        full_model.fit(Xs_train, ys_train, shuffle=True, verbose=3)

        # Train discriminator
        discriminator.trainable = True
        Xs_train, ys_train = prepare_discriminator_batch(generator, X_train, batch_size, .5)
        discriminator.fit(Xs_train, ys_train, shuffle=True, verbose=3)

        Xs_test, ys_test = prepare_discriminator_batch(generator, X_test, 1024, 1)
        loss_real, acc_real = discriminator.evaluate(Xs_test, ys_test, verbose=3)
        Xs_test, ys_test = prepare_discriminator_batch(generator, X_test, 1024, 0)
        loss_fake, acc_fake = discriminator.evaluate(Xs_test, ys_test, verbose=3)
        with open(PROGRESS_FNAME, 'a') as f:
            f.write("%f,%f,%f,%f\n" % (acc_real, acc_fake, loss_real, loss_fake))


        # Clear up memory leaks
        gc.collect()
        tf.keras.backend.clear_session()

        # Polt some generated samples
        view_fake = generator.predict(view_latent, verbose=3)
        samples = np.vstack((view_real, view_fake))
        y = np.vstack((np.ones((10, 1)), np.zeros((10, 1))))
        predictions = discriminator.predict(samples, verbose=3)
        plt.figure(figsize=(20, 5))
        for i in range(10):
            ax = plt.subplot(2, 10, i+1)
            img = samples[i]
            ax.imshow(img, cmap='gray')
            ax.set_xlabel("%.2f (%d)" % (predictions[i][0], y[i][0]))
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = plt.subplot(2, 10, i+11)
            img = samples[i+10]
            ax.imshow(img, cmap='gray')
            ax.set_xlabel("%.2f (%d)" % (predictions[i+10][0], y[i+10][0]))
            ax.set_xticks([])
            ax.set_yticks([])

        plt.savefig("tmp.png")
        plt.close()

        if batch_idx % 25 == 0:
            print("Saving checkpoint models...")
            generator.save("generator.h5")
            discriminator.save("discriminator.h5")

        batch_idx += 1

def combine_models(generator, discriminator):
    inputs = layers.Input(shape=(LATENT_DIM,))

    x = generator(inputs)
    outputs = discriminator(x)

    model = keras.Model(inputs, outputs)

    model.summary()

    # opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    opt = keras.optimizers.Adam(0.00007, 0.6)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def build_generator():

    leakyrelu_alpha = 0.2

    inputs = layers.Input(shape=(LATENT_DIM,))

    d = layers.Dense(512)(inputs)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)
    d = layers.Dense(1024)(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)
    d = layers.Dense(128*8*8)(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)
    d = layers.Reshape((8,8,128))(d)
    
    d = layers.Conv2DTranspose(128, kernel_size=(2,2),  strides=(2,2) , use_bias=False)(d)       # 16,16
    d = layers.LeakyReLU(leakyrelu_alpha)(d)
    d = layers.Conv2D(64, (2, 2), padding='same')(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)


    d = layers.Conv2DTranspose(32, kernel_size=(2,2), strides=(2,2) , use_bias=False)(d)        # 32,32
    d = layers.LeakyReLU(leakyrelu_alpha)(d)
    d = layers.Conv2D(64, (2, 2), padding='same')(d)
    d = layers.LeakyReLU(leakyrelu_alpha)(d)
    
    img = layers.Conv2D(3, (2, 2), activation='sigmoid', padding='same')(d)                     # 32,32
    model = keras.Model(inputs, img)
    model.summary() 

    return model

def build_discriminator():
    leaky_relualpha = .01


    inputs = layers.Input(shape=IM_SHAPE)


    x = layers.Conv2D(32, (3, 3), padding='same', name='block1_conv1')(inputs)
    x = layers.LeakyReLU(leaky_relualpha)(x)
    x = layers.Conv2D(32, (3, 3), padding='same', name='block1_conv2')(x)
    x = layers.LeakyReLU(leaky_relualpha)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', name='block2_conv1')(x)
    x = layers.LeakyReLU(leaky_relualpha)(x)
    x = layers.Conv2D(64, (3, 3), padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = layers.LeakyReLU(leaky_relualpha)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', name='block4_conv1')(x)
    x = layers.LeakyReLU(leaky_relualpha)(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name='block4_conv2')(x)
    x = layers.LeakyReLU(leaky_relualpha)(x)
    x = layers.MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool')(x)
    x = layers.LeakyReLU(leaky_relualpha)(x)


    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model= keras.Model(inputs, out)

    model.summary()
    
    opt = keras.optimizers.Adam(0.00007, 0.6)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", '--load_checkpoints', action="store_true", dest='load_checkpoints')
    args = parser.parse_args()

    train_pool, test_pool = load_data_pool('../../data/Project/celeba_cropped_labels.csv', '../../data/Project/celebA_cropped/')

    if args.load_checkpoints:
        discriminator = keras.models.load_model("discriminator.h5")
        generator = keras.models.load_model("generator.h5")
    else:
        discriminator = build_discriminator()
        generator = build_generator()
        with open(PROGRESS_FNAME, 'w') as f:
            f.write('acc_real,acc_fake,loss_real,loss_fake\n')


    if not args.load_checkpoints:
        pretrian_discriminator(discriminator, 8192, train_pool, test_pool)

    train_models(train_pool, test_pool, discriminator, generator)

if __name__ == "__main__":
    main()