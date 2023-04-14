import argparse
import gc

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

IM_SHAPE = (28, 28, 1)
LATENT_DIM = 100
N_CLASSES = 10

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

def idx_to_one_hot(idxes):
    one_hot = np.eye(N_CLASSES)[idxes]
    return one_hot

def get_training_batch(X_img_all, X_label_all, n, generator = None):
    n = n // 4

    # Get samples that are real and labeled correctly
    idxes = np.random.randint(0, X_img_all.shape[0], n*2)
    X_img = X_img_all[idxes]
    X_label = X_label_all[idxes]

    ys = np.ones((n*2, 1)).reshape((n*2, 1))

    # Get samples that are real but labeled incorrectly
    idxes = np.random.randint(0, X_img_all.shape[0], n)
    X_img_temp = X_img_all[idxes]
    X_label_temp = (X_label_all[idxes] + np.random.randint(1, 10, n)) % 10

    X_img = np.vstack((X_img, X_img_temp))
    X_label = np.hstack((X_label, X_label_temp))
    ys = np.vstack((ys, np.zeros((n, 1))))


    # Get samples that are "generated"
    if generator is None:
        gen_samples_train = get_random_samples(n)
        gen_labels_train = np.random.randint(0, 10, n)
    else:
        lat_inputs = get_latent_space_inputs(n)
        gen_labels_train = np.random.randint(0, 10, n)
        gen_labels_train_one_hot = idx_to_one_hot(gen_labels_train)
        gen_samples_train = generator.predict((lat_inputs, gen_labels_train_one_hot), verbose=3).reshape((n, IM_SHAPE[0], IM_SHAPE[1]))

    X_img = np.vstack([X_img, gen_samples_train])
    X_label = np.hstack([X_label, gen_labels_train])
    X_label = idx_to_one_hot(X_label)

    ys = np.vstack((ys, np.zeros((n, 1))))

    return X_img, X_label, ys

def pretrian_discriminator(discriminator, n, X_img_train, X_label_train, X_img_test, X_label_test):

    X_img_train, X_label_train, ys_train = get_training_batch(X_img_train, X_label_train, n)
    X_img_test, X_label_test, ys_test = get_training_batch(X_img_test, X_label_test, n // 2)

    print("Training initial discriminator...")
    discriminator.fit((X_img_train, X_label_train), ys_train, epochs=5, batch_size=8, shuffle=True, validation_data=((X_img_test, X_label_test), ys_test))
    print("Done!")

def train_models(X_img_train, X_label_train, X_img_test, X_label_test, discriminator, generator, batch_size = 1024):

    # 1 = real, 0 = fake

    full_model = combine_models(generator, discriminator)

    draw_rows = 10
    draw_latent_inputs = get_latent_space_inputs(10 * draw_rows)
    draw_label_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * draw_rows)
    draw_label_inputs = idx_to_one_hot(draw_label_inputs)

    print("Starting training...")
    batch_idx = 0
    while True:

        # Train generator
        discriminator.trainable = False
        generator.trainable = True

        lat_inputs = get_latent_space_inputs(batch_size)
        label_inputs = np.random.randint(0, 10, batch_size)
        label_inputs = idx_to_one_hot(label_inputs)
        ys = np.ones((batch_size, 1))

        full_model.fit((lat_inputs, label_inputs), ys, epochs=1, batch_size=32, verbose = 3, shuffle=True)

        # Train discriminator
        discriminator.trainable = True

        Xs_img, Xs_labels, ys = get_training_batch(X_img_train, X_label_train, batch_size, generator)

        discriminator.fit((Xs_img, Xs_labels), ys, epochs=1, batch_size=32, shuffle=True, verbose=3)

        # Evaluate models
        Xs_img_test, Xs_labels_test, ys_test = get_training_batch(X_img_test, X_label_test, batch_size, generator)

        loss, acc = discriminator.evaluate((Xs_img_test, Xs_labels_test), ys_test, batch_size=32, verbose=3)
        print("Discriminator loss: %.2f, accuracy: %.2f" % (loss, acc))
        loss, acc = full_model.evaluate( (get_latent_space_inputs(batch_size), idx_to_one_hot(np.random.randint(0, 10, batch_size)) ), np.ones((Xs_img_test.shape[0], 1)), batch_size=32, verbose=3)
        print("Generator loss: %.2f, accuracy: %.2f" % (loss, acc))
        print("====================================")
        # Clear up memory leaks
        # gc.collect()
        # tf.keras.backend.clear_session()

        # Polt some generated samples
        generated_imgs = generator.predict((draw_latent_inputs, draw_label_inputs), verbose=3).reshape((10*draw_rows, IM_SHAPE[0], IM_SHAPE[1]))
        plt.figure(figsize=(10, 10))
        for i in range(10):
            for j in range(draw_rows):
                ax = plt.subplot(draw_rows, 10, i + j * 10 + 1)
                ax.imshow(generated_imgs[i + j * 10], cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if j == 0:
                    ax.set_title("%d" % i)

            

        plt.subplots_adjust(left=0, right=1, top=0.962, bottom=0, wspace=0, hspace=0.1)
        plt.savefig("tmp.png")
        plt.close()

        if batch_idx % 50 == 0:
            print("Saving checkpoint models...")
            generator.save("generator.h5")
            discriminator.save("discriminator.h5")

        batch_idx += 1

def combine_models(generator, discriminator):
    latent_inputs = layers.Input(shape=(LATENT_DIM,))
    class_input = layers.Input(shape=(N_CLASSES,))

    x = generator((latent_inputs, class_input))

    outputs = discriminator((x, class_input))

    model = keras.Model((latent_inputs, class_input), outputs)

    model.summary()

    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def build_generator():
    inputs_latent = layers.Input(shape=(LATENT_DIM,))
    inputs_class = layers.Input(shape=(N_CLASSES,))

    inputs = layers.Concatenate()([inputs_latent, inputs_class])

    x = layers.Dense(128)(inputs)
    skip = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(128)(skip)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Concatenate()([x, skip])
    x = layers.Dense(128 * 7 * 7)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Reshape((7, 7, 128))(x)
    
    # Upsample to 14x14
    x = layers.Conv2DTranspose(256, (4, 4), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Upsample to 28x28
    x = layers.Conv2DTranspose(256, (4, 4), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)


    x = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(x)

    model = keras.Model((inputs_latent, inputs_class), x)

    model.summary()

    return model

def build_discriminator():
    class_inputs = layers.Input(shape=(N_CLASSES,))
    img_inputs = layers.Input(shape=IM_SHAPE)

    x = layers.Conv2D(64, (3, 3), padding='same')(img_inputs)
    skip = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(skip)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Concatenate()([x, skip])
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)


    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    skip = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(skip)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Concatenate()([x, skip])
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)


    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, class_inputs])
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(16)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model([img_inputs, class_inputs], x)

    model.summary()
    
    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
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
        pretrian_discriminator(discriminator, 8192*2, X_train, y_train, X_test, y_test)

    train_models(X_train, y_train, X_test, y_test, discriminator, generator)

if __name__ == "__main__":
    main()