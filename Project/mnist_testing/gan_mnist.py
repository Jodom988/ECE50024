import argparse
import gc

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

IM_SHAPE = (28, 28, 1)
LATENT_DIM = 128

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)

def get_random_samples(n):
    samples = list()
    for _ in range(n):
        sample = np.random.rand(IM_SHAPE[0], IM_SHAPE[1])
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
    # First, train the discriminator on real and fake images (give it a head start)
    rand_samples_train = get_random_samples(n)
    X_train = X_train[:n]

    rand_samples_test = get_random_samples(n)
    X_test = X_test[:n]

    Xs_train = np.vstack((rand_samples_train, X_train))
    ys_train = np.vstack((np.zeros((n, 1)), np.ones((n, 1))))

    Xs_test = np.vstack((rand_samples_test, X_test))
    ys_test = np.vstack((np.zeros((n, 1)), np.ones((n, 1))))

    print("Training initial discriminator...")
    print(Xs_train.shape, ys_train.shape)
    print(Xs_test.shape, ys_test.shape)
    discriminator.fit(Xs_train, ys_train, epochs=2, batch_size=128, validation_data=(Xs_test, ys_test), shuffle=True)
    print("Done!")

def train_models(X_train, X_test, discriminator, generator, batch_size = 64):

    # 1 = real, 0 = fake

    full_model = combine_models(generator, discriminator)

    print("Starting training...")
    batch_idx = 0
    while True:

        # Train generator
        discriminator.trainable = False
        generator.trainable = True

        Xs_train = get_latent_space_inputs(batch_size)
        ys_train = np.ones((batch_size, 1))

        full_model.fit(Xs_train, ys_train, epochs=1, batch_size=32, verbose = 3, shuffle=True)

        # Train discriminator
        discriminator.trainable = True

        generated_samples = generator.predict(get_latent_space_inputs(batch_size // 2)).reshape((batch_size // 2, IM_SHAPE[0], IM_SHAPE[1]))
        np.random.shuffle(X_train)
        real_samples = X_train[:batch_size // 2, :, :]
        Xs_train = np.vstack((generated_samples, real_samples))
        ys_train = np.vstack((np.zeros((batch_size // 2, 1)), np.ones((batch_size // 2, 1))))

        generated_samples = generator.predict(get_latent_space_inputs(batch_size // 2)).reshape((batch_size // 2, IM_SHAPE[0], IM_SHAPE[1]))
        np.random.shuffle(X_test)
        real_samples = X_test[:batch_size // 2, :, :]
        Xs_test = np.vstack((generated_samples, real_samples))
        ys_test = np.vstack((np.zeros((batch_size // 2, 1)), np.ones((batch_size // 2, 1))))

        discriminator.fit(Xs_train, ys_train, epochs=1, batch_size=32, validation_data=(Xs_test, ys_test), shuffle=True)

        # Clear up memory leaks
        # gc.collect()
        # tf.keras.backend.clear_session()

        # Polt some generated samples
        plt.figure(figsize=(20, 4))
        for i in range(10):
            ax = plt.subplot(2, 10, i+1)
            ax.imshow(generated_samples[i], cmap='gray')
            
            ax = plt.subplot(2, 10, i+11)
            ax.imshow(generated_samples[i+10], cmap='gray')

        plt.savefig("tmp.png")
        plt.close()

        if batch_idx % 50 == 0:
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

    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def build_generator():
    inputs = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense(128 * 7 * 7)(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Reshape((7, 7, 128))(x)
    
    # Upsample to 14x14
    x = layers.Conv2DTranspose(256, (4, 4), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Upsample to 28x28
    x = layers.Conv2DTranspose(256, (4, 4), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(x)

    model = keras.Model(inputs, x)

    model.summary()

    return model

def build_discriminator():
    inputs = layers.Input(shape=IM_SHAPE)

    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.05)(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.05)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)


    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.05)(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.05)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)


    x = layers.Flatten()(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.05)(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, x)

    model.summary()
    
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", '--load_checkpoints', action="store_true", dest='load_checkpoints')
    args = parser.parse_args()

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    if args.load_checkpoints:
        discriminator = keras.models.load_model("discriminator.h5")
        generator = keras.models.load_model("generator.h5")
    else:
        discriminator = build_discriminator()
        generator = build_generator()
        pretrian_discriminator(discriminator, 8192, X_train, X_test)

    train_models(X_train, X_test, discriminator, generator)

if __name__ == "__main__":
    main()