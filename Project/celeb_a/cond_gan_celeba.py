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

MALE_IDX = 21
EYEGLASSES_IDX = 16
SMILING_IDX = 32

IM_SHAPE = (32, 32, 3)
LATENT_SHAPE = (8, 8, 128)
FILTER_LABELS = [EYEGLASSES_IDX]
FILTER_LABELS_VALS = [0]
LABEL_IDXS = [MALE_IDX, SMILING_IDX]
N_LABELS = 2 * len(LABEL_IDXS)
NOISE_DIM = 100-N_LABELS
CHECKPOINT_DIR = 'gan_labels'
PROGRESS_POST_GEN_FNAME = 'progress_post_generator.csv'
PROGRESS_POST_DISC_FNAME = 'progress_post_discriminator.csv'

PAST_GEN_POOL_MAX = 50000

LEARNING_RATE = 0.00007
# LEARNING_RATE = 0.0001
# LEARNING_RATE = 0.0005

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)


class GAN():

    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

        input_noise = layers.Input(shape=(NOISE_DIM,))
        input_label = layers.Input(shape=(N_LABELS,))
        input_img = layers.Input(shape=IM_SHAPE)

        # Create training discriminator (add noise and dropout)
        # x = layers.GaussianNoise(.01)(input_img)
        x = layers.Dropout(0.2)(input_img)
        output = self.discriminator((x, input_label))
        self.training_discriminator = keras.Model(inputs=[input_img, input_label], outputs=output, name='training_discriminator')
        opt = keras.optimizers.Adam(LEARNING_RATE, 0.5)
        self.training_discriminator.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'mean_absolute_error'])

        # Create full training model
        x = self.generator((input_noise, input_label))
        output = self.discriminator((x, input_label))
        self.full_model = keras.Model(inputs=[input_noise, input_label], outputs=output, name='full_model')
        opt = keras.optimizers.Adam(LEARNING_RATE, 0.5)
        self.full_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'mean_absolute_error'])


    def train_generator(self, Xs, ys, batch_size=32, verbose=3, shuffle=True):
        self.generator.trainable = True
        self.discriminator.trainable = False

        h = self.full_model.fit(Xs, ys, batch_size=batch_size, verbose=verbose, shuffle=shuffle)
        return h.history['accuracy'][0], h.history['mean_absolute_error'][0]

    def train_discriminator(self, Xs, ys, batch_size=32, verbose=3, validation_data=None, shuffle=True):
        self.discriminator.trainable = True

        h = self.training_discriminator.fit(Xs, ys, batch_size=batch_size, verbose=verbose, validation_data=validation_data, shuffle=shuffle)
        return h.history['accuracy'][0], h.history['mean_absolute_error'][0]

    def generate_full_img(self, latent, label, verbose=3):
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



def load_data_pool(csv_fpath, img_dir, pct_train=0.8):

    with open(csv_fpath, 'r') as f:
        lines = f.readlines()

    # lines = lines[:len(lines)//10]

    data = list()
    i = 0
    for line in tqdm(lines, desc="Loading data"):
        if i == 0:
            i += 1
            continue
        
        line = line.split(',')

        bad_sample = False
        for j, idx in enumerate(FILTER_LABELS):
            if int(line[idx]) != FILTER_LABELS_VALS[j]:
                bad_sample = True
                break
        
        if bad_sample:
            i += 1
            continue
        
        # print("Found good sample: %d" % i)

        img_name = line[0]
        img_path = os.path.join(img_dir, img_name)

        embeddings = list()
        for idx in LABEL_IDXS:
            embeddings.append(float(line[idx]))
            embeddings.append(1.0 - float(line[idx]))
        embeddings = np.array(embeddings)

        data.append((img_path, embeddings))
        i += 1

    np.random.seed(123)
    np.random.shuffle(data)

    idx = int(len(data) * pct_train)
    train_data = data[:idx]
    test_data = data[idx:]
    return train_data, test_data

def load_imgs(data_pool):
    img_fpaths = [data_pool[i][0] for i in range(len(data_pool))]

    imgs = list()
    for img_fpath in tqdm(img_fpaths, desc="Reading images from files..."):
        img = Image.open(img_fpath)
        img = img.resize((IM_SHAPE[0], IM_SHAPE[1]))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(IM_SHAPE)
        imgs.append(img)

    new_data_pool = [(imgs[i], data_pool[i][1]) for i in range(len(data_pool))]
    return new_data_pool

def get_discriminator_batch(gan: GAN, data_pool, n, w_real=1.0, w_fake=1.0, w_swapped=1.0):
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
        idxes = np.random.choice(list(range(len(data_pool))), n_real, replace=False)
        X_real_imgs, X_real_labels = imgs_from_idxes(data_pool, idxes)
        ys = np.ones((n_real, 1))

        X_imgs = X_real_imgs
        X_labels = X_real_labels

    # get generated samples
    if n_fake != 0:
        if gan is None:
            X_fake_imgs, X_fake_labels = get_random_samples(n_fake)
        else:
            noise_inputs, X_fake_labels = get_noise_inputs(n_fake, data_pool)
            X_fake_imgs = gan.generator.predict((noise_inputs, X_fake_labels), verbose=3)

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
        X_swapped_imgs, X_swapped_labels = imgs_from_idxes(data_pool, idxes)
        
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

def imgs_from_idxes(data_pool, idexs):
    batch_imgs = list()
    batch_embeddings = list()
    for idx in idexs:
        batch_imgs.append(data_pool[idx][0])

        embeddings = data_pool[idx][1]
        embeddings = np.array(embeddings)
        batch_embeddings.append(embeddings)

    batch_imgs = np.array(batch_imgs)
    batch_embeddings = np.array(batch_embeddings)
    return batch_imgs, batch_embeddings

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

def get_noise_inputs(n, data_pool):
    inputs = list()
    labels = list()

    idxes = np.random.choice(list(range(len(data_pool))), n, replace=False)

    for i in range(n):
        input = np.random.normal(0, 0.5, NOISE_DIM)
        inputs.append(input)

        label = data_pool[idxes[i]][1]
        labels.append(label)

    inputs = np.array(inputs)
    labels = np.array(labels)

    return inputs, labels

def pretrian_discriminator(gan: GAN, n, X_train, X_test):
    
    n_train = n
    n_test = n // 4
    Xs_test, ys_test = get_discriminator_batch(None, X_test, n_test, w_swapped=0)

    for epoch in range(5):
        Xs_train, ys_train = get_discriminator_batch(None, X_train, n_train, w_swapped=0)

        print("Training initial discriminator...")
        gan.train_discriminator(Xs_train, ys_train, validation_data=(Xs_test, ys_test), verbose=1)
        print("Done!")

def evaluate_models(gan: GAN, X_test, epoch_pct, data_idx, fname):
    Xs_test, ys_test = get_discriminator_batch(gan, X_test, 2048, w_real=1.0, w_fake=0, w_swapped=0)
    loss_real, acc_real, mae_real = gan.discriminator.evaluate(Xs_test, ys_test, verbose=3)
    avg_confidence_real = np.mean(gan.discriminator.predict(Xs_test, verbose=3))
    # avg_confidence_real = 1 - mae_real

    # print("============================")
    # print(avg_confidence_real)
    # print(1 - mae_real)
    # print("============================")

    Xs_test, ys_test = get_discriminator_batch(gan, X_test, 2048, w_fake=1.0, w_real=0, w_swapped=0)
    loss_fake, acc_fake, mae_fake = gan.discriminator.evaluate(Xs_test, ys_test, verbose=3)
    avg_confidence_fake = 1 - np.mean(gan.discriminator.predict(Xs_test, verbose=3))

    # print("============================")
    # print(avg_confidence_fake)
    # print(mae_fake)
    # print("============================")

    with open(os.path.join(CHECKPOINT_DIR, fname), 'a') as f:
        f.write("%f,%d,%f,%f,%f,%f,%f,%f\n" % (epoch_pct, data_idx, acc_real, acc_fake, loss_real, loss_fake, avg_confidence_real, avg_confidence_fake))


def add_past_gen_pool(past_gen_pool, new_additions):
    past_gen_latent = past_gen_pool[0]
    past_gen_labels = past_gen_pool[1]

    if len(past_gen_pool) < PAST_GEN_POOL_MAX:
        past_gen_latent = np.vstack((past_gen_latent, new_additions[0]))
        past_gen_labels = np.vstack((past_gen_labels, new_additions[1]))
    else:
        ratio_add = .5
        num_add = int(new_additions[0].shape[0] * ratio_add)
        idxes = np.random.choice(list(range(PAST_GEN_POOL_MAX)), num_add, replace=False)

        past_gen_latent[idxes] = new_additions[0][:num_add]
        past_gen_labels[idxes] = new_additions[1][:num_add]

    return (past_gen_latent, past_gen_labels)

def extend_batch_with_past_gen(batch_xs, batch_ys, past_gen_pool, n, gan: GAN):
    batch_latent = batch_xs[0]
    batch_labels = batch_xs[1]

    past_gen_pool_len = past_gen_pool[0].shape[0]
    n = min(n, past_gen_pool_len - 1)

    idxes = np.random.choice(list(range(past_gen_pool[0].shape[0])), n, replace=False)
    past_gen_latent = past_gen_pool[0][idxes]
    past_gen_labels = past_gen_pool[1][idxes]

    # plt.subplot(10, 10, 1)
    # imgs = gan.decoder.predict(past_gen_latent[:100], verbose=3)
    # for i in range(100):
    #     plt.subplot(10, 10, i+1)
    #     plt.imshow(imgs[i].reshape(IM_SHAPE))
    #     plt.axis('off')

    # plt.show()

    batch_latent = np.vstack((batch_latent, past_gen_latent))
    batch_labels = np.vstack((batch_labels, past_gen_labels))
    batch_ys = np.vstack((batch_ys, np.zeros((n, 1))))

    return (batch_latent, batch_labels), batch_ys


def train_models(X_train, X_test, gan: GAN, init_data_idx, batch_size = 256):

    np.random.seed(2)
    view_real, _ = get_discriminator_batch(None, X_test, 10, w_real=1.0, w_fake=0, w_swapped=0)
    view_noise = get_noise_inputs(10, X_test)

    test_gen_Xs = get_noise_inputs(64, X_test)
    test_gen_ys = np.ones((64, 1))

    past_gens = get_random_samples(batch_size)
    print("===== INIT PAST_GEN SIZE =====")
    print(past_gens[0].shape)

    print("Starting training...")
    batch_idx = 0
    data_idx = init_data_idx
    np.random.seed(int(time.time()))
    batches_at_once = 4
    while True:

        # Train generator
        mae = 1
        count = 0
        # print("Generator")
        while mae > 0.1 and count < 128:
            Xs_train = get_noise_inputs(batch_size*batches_at_once, X_train)
            ys_train = np.ones((batch_size*batches_at_once, 1))

            acc, mae = gan.train_generator(Xs_train, ys_train, batch_size=batch_size, verbose=3)
            count += batches_at_once

            loss, acc, mae = gan.full_model.evaluate(test_gen_Xs, test_gen_ys, verbose=3)

        print("Generator:     %d, %.2f, %.2f" % (count, acc, mae))

        # Evaluate post generator training
        if batch_idx % 5 == 0:
            evaluate_models(gan, X_test, data_idx / len(X_train), data_idx, PROGRESS_POST_GEN_FNAME)

        # Train discriminator
        mae = 1
        count = 0
        test_disc_Xs, test_disc_ys = get_discriminator_batch(gan, X_test, 64, w_swapped=0)
        # print("Discriminator")

        disc_mult = 4
        while mae > 0.1 and count < 128:
            Xs_train, ys_train = get_discriminator_batch(gan, X_train, batch_size*batches_at_once*disc_mult, w_swapped=0)
            Xs_train, ys_train = extend_batch_with_past_gen(Xs_train, ys_train, past_gens, batch_size*batches_at_once*disc_mult//2, gan)

            acc, mae = gan.train_discriminator(Xs_train, ys_train, batch_size=batch_size, verbose=3)
            loss, acc, mae = gan.discriminator.evaluate(test_disc_Xs, test_disc_ys, verbose=3)
            data_idx += batch_size
            count += batches_at_once*disc_mult

        noise, labels = get_noise_inputs(batch_size//2, X_train)
        past_gens = add_past_gen_pool(past_gens, (gan.generator((noise, labels)), labels))

        print("Discriminator: %d, %.2f, %.2f" % (count, acc, mae))

        # Clear up memory leaks
        gc.collect()
        tf.keras.backend.clear_session()

        # Evaluate post discriminator training
        if batch_idx % 5 == 0:
            evaluate_models(gan, X_test, data_idx / len(X_train), data_idx, PROGRESS_POST_DISC_FNAME)

        if batch_idx % 2 == 0:
            print("Updating view")
            # Polt some generated samples
            gnerated_imgs = gan.generator.predict(view_noise, verbose=3)
            view_imgs = np.vstack((view_real[0], gnerated_imgs))
            view_labels = np.vstack((view_real[1], view_noise[1]))
            predictions = gan.discriminator.predict((view_imgs, view_labels), verbose=3)
            y = np.vstack((np.ones((10, 1)), np.zeros((10, 1))))

            plt.figure(figsize=(20, 5))
            for i in range(10):
                ax = plt.subplot(2, 10, i+1)
                img = view_imgs[i]
                ax.imshow(img)
                class_vals = [view_real[1][i][j] for j in range(N_LABELS)]
                class_vals = ",".join([str(int(x)) for x in class_vals])
                ax.set_xlabel("%.2f (%d)\n%s" % (predictions[i][0], y[i][0], class_vals))
                ax.set_xticks([])
                ax.set_yticks([])
                
                ax = plt.subplot(2, 10, i+11)
                img = view_imgs[i+10]
                ax.imshow(img)
                class_vals = [view_noise[1][i][j] for j in range(N_LABELS)]
                class_vals = ",".join([str(int(x)) for x in class_vals])
                ax.set_xlabel("%.2f (%d)\n%s" % (predictions[i+10][0], y[i+10][0], class_vals))
                ax.set_xticks([])
                ax.set_yticks([])

            plt.savefig(os.path.join(CHECKPOINT_DIR, "tmp.png"))
            plt.close()
            plt.clf()
        
        if batch_idx % 5 == 0:
            print("Saving checkpoint models")
            gan.save(CHECKPOINT_DIR)
            print("Done")

        batch_idx += 1

def build_generator():

    inputs_noise = layers.Input(shape=(NOISE_DIM,))
    inputs_labels = layers.Input(shape=(N_LABELS,))
    inputs = layers.concatenate([inputs_noise, inputs_labels])

    leakyrely_alpha = 0.01

    d = layers.Dense(256)(inputs)
    d = layers.LeakyReLU(leakyrely_alpha)(d)
    d = layers.BatchNormalization()(d)

    d = layers.Dense(512)(inputs)
    d = layers.LeakyReLU(leakyrely_alpha)(d)
    d = layers.BatchNormalization()(d)

    d = layers.Dense(2*2*1024)(inputs)
    d = layers.LeakyReLU(leakyrely_alpha)(d)
    d = layers.BatchNormalization()(d)

    d = layers.Reshape((2,2,1024))(d)

    d = layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(d)
    d = layers.LeakyReLU(leakyrely_alpha)(d)
    d = layers.BatchNormalization()(d)

    d = layers.Conv2D(512, (3,3), padding='same')(d)
    d = layers.LeakyReLU(leakyrely_alpha)(d)
    d = layers.BatchNormalization()(d)

    d = layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(d)
    d = layers.LeakyReLU(leakyrely_alpha)(d)
    d = layers.BatchNormalization()(d)

    d = layers.Conv2D(256, (3,3), padding='same')(d)
    d = layers.LeakyReLU(leakyrely_alpha)(d)
    d = layers.BatchNormalization()(d)

    d = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(d)
    d = layers.LeakyReLU(leakyrely_alpha)(d)
    d = layers.BatchNormalization()(d)

    d = layers.Conv2D(128, (3,3), padding='same')(d)
    d = layers.LeakyReLU(leakyrely_alpha)(d)
    d = layers.BatchNormalization()(d)

    d = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(d)
    d = layers.LeakyReLU(leakyrely_alpha)(d)
    d = layers.BatchNormalization()(d)

    d = layers.Conv2D(128, (3,3), padding='same')(d)
    d = layers.LeakyReLU(leakyrely_alpha)(d)
    
    d = layers.Conv2D(3, (7,7), padding='same', activation='sigmoid')(d)

    # d = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(d)

    
    model = keras.Model((inputs_noise, inputs_labels), d, name="latent_generator_model")
    model.summary()

    opt = keras.optimizers.Adam(LEARNING_RATE, 0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'mean_absolute_error'])

    return model

def build_discriminator():
    leakyrelu_alpha = 0.01


    inputs_img = layers.Input(shape=(32, 32, 3))
    inputs_labels = layers.Input(shape=(N_LABELS,))

    x = layers.Conv2D(128, (2,2), padding='same')(inputs_img)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(256, (2,2), padding='same')(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(512, (2,2), padding='same')(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(512, (2,2), padding='same')(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, inputs_labels])

    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.Dense(128)(x)
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
    parser.add_argument("-d", '--load_discriminator', action="store_true", dest='load_discriminator')
    parser.add_argument("-g", '--load_generator', action="store_true", dest='load_generator')
    args = parser.parse_args()



    if args.load_checkpoints:
        gan = GAN.load(CHECKPOINT_DIR)

        df = pd.read_csv(os.path.join(CHECKPOINT_DIR, PROGRESS_POST_DISC_FNAME))
        data_idx = df['data_idx'].iloc[-1]
    else:

        if args.load_discriminator:
            discriminator = keras.models.load_model(os.path.join(CHECKPOINT_DIR, 'discriminator.h5'))
        else:
            discriminator = build_discriminator()
        
        if args.load_generator:
            generator = keras.models.load_model(os.path.join(CHECKPOINT_DIR, 'generator.h5'))
        else:
            generator = build_generator()

        gan = GAN(generator, discriminator)

        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
        os.makedirs(CHECKPOINT_DIR)

        with open(os.path.join(CHECKPOINT_DIR, PROGRESS_POST_DISC_FNAME), 'w') as f:
            f.write('epoch_pct,data_idx,acc_real,acc_fake,loss_real,loss_fake,avg_confidence_real,avg_confidence_fake\n')
        
        with open(os.path.join(CHECKPOINT_DIR, PROGRESS_POST_GEN_FNAME), 'w') as f:
            f.write('epoch_pct,data_idx,acc_real,acc_fake,loss_real,loss_fake,avg_confidence_real,avg_confidence_fake\n')
        data_idx = 0

    train_pool, test_pool = load_data_pool('../../data/Project/celeba_cropped_labels.csv', '../../data/Project/celebA_cropped/')
    train_pool = load_imgs(train_pool)
    test_pool = load_imgs(test_pool)

    if not args.load_checkpoints:
        pretrian_discriminator(gan, 8192, train_pool, test_pool)

    train_models(train_pool, test_pool, gan, data_idx)

if __name__ == "__main__":
    main()