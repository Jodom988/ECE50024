import argparse
import os
import threading
from queue import Queue
import gc

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm import tqdm
import absl.logging

from common import label_name_mapping, read_label_csv

import random

IM_SIZE = (128, 128, 1)
MODEL_BASENAME = "naive"

random.seed(1)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)

absl.logging.set_verbosity(absl.logging.ERROR)

def load_normalize_img(path):
    if (os.path.exists(path) == False):
        print("ERROR: Path does not exist: " + path)
        return None

    img = Image.open(path)
    img = ImageOps.grayscale(img)
    width, height = img.size

    if width < height:
        new_height = IM_SIZE[1]
        new_width = int(width * (new_height / height))
        img = img.resize((new_width, new_height))
    else:
        new_width = IM_SIZE[0]
        new_height = int(height * (new_width / width))
        img = img.resize((new_width, new_height))

    img = ImageOps.expand(img, border=(0, 0, IM_SIZE[0] - new_width, IM_SIZE[1] - new_height), fill=0)

    # save image
    # img.save("tmp.png")

    # img.show()
    # input()

    return img

def load_imgs(data):

    imgs = list()
    labels = list()

    for i in tqdm(range(len(data)), desc="Loading images"):
        path, class_label, name = data[i]
        img = load_normalize_img(path)
        if img is None:
            continue

        img = np.array(img)
        imgs.append(img)
        label = np.zeros(100)
        label[class_label] = 1
        labels.append(label)
        # labels.append(class_label)

    imgs = np.array(imgs)
    labels = np.array(labels)
    return imgs, labels

def load_batch(batch_size, data, idxes):
    imgs = list()
    labels = list()
    # print("Loading batch with idxes: %s" % str(idxes))
    for i in idxes:
        path, class_label, name = data[i]
        img = load_normalize_img(path)
        if img is None:
            continue

        img = np.array(img)
        imgs.append(img)
        label = np.zeros(100)
        label[class_label] = 1
        labels.append(label)
        # labels.append(class_label)

    imgs = np.array(imgs)
    labels = np.array(labels)

    return imgs, labels

def load_imgs_into_new_img_queue(q: Queue, data, stop_event, start_idx = 0):
    curr_idx = start_idx
    while True:
        batch_len = min(64, len(data) - curr_idx)
        idxes = range(curr_idx, curr_idx + batch_len)
        imgs, labels = load_batch(64, data, idxes)
        curr_idx += batch_len
        if curr_idx >= len(data):
            curr_idx = 0
        if len(imgs) != len(labels):
            print("ERROR: len(imgs) != len(labels)")
            continue
        for i in range(len(imgs)):
            q.put((imgs[i], labels[i]))
            if stop_event.is_set():
                return

def load_imgs_into_old_imgs_queue(q: Queue, data, stop_event, max_idx: list):
    while True:
        idxes = list()
        last = min(max_idx[0], len(data) - 1)
        for i in range(64):
            idxes.append(random.randint(0, last))

        imgs, labels = load_batch(len(idxes), data, idxes)
        
        if len(imgs) != len(labels):
            print("ERROR: len(imgs) != len(labels)")
            continue

        for i in range(len(imgs)):
            q.put((imgs[i], labels[i]))
            if stop_event.is_set():
                return

def load_imgs_from_queue(q: Queue, num_imgs):

    imgs = list()
    labels = list()

    for _ in range(num_imgs):
        img, label = q.get()
        imgs.append(img)
        labels.append(label)

    return imgs, labels

def update_checkpoint(model, data_idx):
    dir = os.path.join("models", "%s" % (MODEL_BASENAME))

    checkpoint_data_fname = os.path.join(dir, "num_data_read.txt")
    with open(checkpoint_data_fname, "w") as f:
        f.write("%d\n" % (data_idx))


    checkpoint_dir = os.path.join(dir, "checkpoint_model")
    model.save(checkpoint_dir)


def train_model(model, train_data, test_data, batch_size = 64, num_batches = 25, data_idx = 0):

    new_img_q = Queue(maxsize=4096)
    stop_new_img_q_event = threading.Event()
    new_imgs_load_thread = threading.Thread(target=load_imgs_into_new_img_queue, args=(new_img_q, train_data, stop_new_img_q_event, data_idx))
    new_imgs_load_thread.start()

    last_idx = [data_idx + 32]
    old_img_q = Queue(maxsize=256)
    stop_old_img_q_event = threading.Event()
    old_imgs_load_thread = threading.Thread(target=load_imgs_into_old_imgs_queue, args=(old_img_q, train_data, stop_old_img_q_event, [len(train_data)]))
    old_imgs_load_thread.start()

    print("Reading test data...")
    test_imgs, test_labels = load_imgs(test_data)
    test_imgs = np.array(test_imgs)
    test_labels = np.array(test_labels)
    
    print("Starting training with batch size %d and %d number of batches" % (batch_size, num_batches))
    if num_batches == -1:
        iterator = range(10000000000000)   # Pseudo Forever
    else:
        iterator = tqdm(range(num_batches))

    try:
        best_acc = 0
        for epoch in iterator:
            # Load new images to train the model on
            train_imgs, train_labels = load_imgs_from_queue(new_img_q, batch_size)
            train_imgs = np.array(train_imgs)
            train_labels = np.array(train_labels)

            model.fit(train_imgs, train_labels, verbose=3)

            # Load old images to train the model on to avoid catestrophic forgetting
            train_imgs, train_labels = load_imgs_from_queue(old_img_q, batch_size)
            train_imgs = np.array(train_imgs)
            train_labels = np.array(train_labels)
            
            model.fit(train_imgs, train_labels, verbose=3)

            # Evaluate the model
            test_loss, test_acc = model.evaluate(test_imgs,  test_labels, verbose=0, batch_size=batch_size)

            # Call this to free up memory
            gc.collect()
            keras.backend.clear_session()

            # Update how much data we've seen
            data_idx += batch_size
            last_idx[0] = min(data_idx, len(train_data))

            # Save the model accordingly and update stats
            if best_acc < test_acc:
                best_acc = test_acc
                dir = os.path.join("models", "%s" % (MODEL_BASENAME))
                dir = os.path.join(dir, "best_model")
                model.save(dir)
                
            fname = os.path.join("models", "%s" % (MODEL_BASENAME))
            fname = os.path.join(fname, "accuracies.txt")
            with open(fname, "a") as f:
                f.write("%f\n" % (test_acc))

            if epoch % 10 == 0:
                update_checkpoint(model, data_idx)

    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting...")
    except Exception as e:
        print("Encounted unexpected exception...")
        update_checkpoint(model, data_idx)
        raise e

    print("Updating checkpoint")
    update_checkpoint(model, data_idx)
    stop_new_img_q_event.set()
    stop_old_img_q_event.set()

    # Close the new image getter thread
    while True:
        print("Attempting join...")
        for _ in range(new_img_q.qsize()):
            new_img_q.get(timeout=.1)
        new_imgs_load_thread.join(timeout=.5)
        if not new_imgs_load_thread.is_alive():
            break
    
    # Close the old image getter thread
    while True:
        print("Attempting join...")
        for _ in range(old_img_q.qsize()):
            old_img_q.get(timeout=.1)
        old_imgs_load_thread.join(timeout=.5)
        if not old_imgs_load_thread.is_alive():
            break

def get_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=IM_SIZE))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))

    model.add(layers.Flatten())
    
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))

    model.summary()

    model.compile(optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_csv', help='CSV with filepaths and labels')
    parser.add_argument('test_csv', help='CSV with filepaths and labels')
    parser.add_argument('img_dir', help='Directory with images')
    parser.add_argument('-b' '--batch_size', help='Batch size for training', default=1024, type=int)
    parser.add_argument('-n' '--num_batches', help='Number of ephocs', default=25, type=int)
    parser.add_argument('-p' '--pct_test_data', help='Percent of test data to use', default=None, type=int)

    args = parser.parse_args()

    train_data = read_label_csv(args.train_csv, args.img_dir)
    test_data = read_label_csv(args.test_csv, args.img_dir)

    if args.p__pct_test_data:
        print("TRIMMING TEST DATA")
        test_data = test_data[:int(len(test_data) * args.p__pct_test_data / 100)]

    if os.path.exists("models/naive/checkpoint_model"):
        model = tf.keras.models.load_model("models/naive/checkpoint_model")
        with open("models/naive/num_data_read.txt", "r") as f:
            data_idx = int(f.read())
            print("Found checkpoint. Resuming training from data index %d" % (data_idx))

    else:    
        model = get_model()
        data_idx = 0

    if not (os.path.exists("models/naive")):
        os.mkdir("models/naive")

    train_model(model, train_data, test_data, batch_size=args.b__batch_size, num_batches=args.n__num_batches, data_idx=data_idx)

if __name__ == '__main__':
    main()