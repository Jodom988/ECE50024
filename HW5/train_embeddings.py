import os
import threading
from queue import Queue
import random
import absl.logging
import argparse
import gc

absl.logging.set_verbosity(absl.logging.ERROR)

from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np

from common import label_name_mapping, read_label_csv

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models

INPUT_VECTOR_SHAPE = (2622)
N_LABELS = 100

MODEL_BASENAME = "deepface_embeddings"

def read_embeddings_csv(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()

    labels = []
    data = []
    for i, line in tqdm(list(enumerate(lines))):
        if i == 0:
            continue
        parts = line.split(',')

        cat = int(parts[0])
        label = np.zeros(N_LABELS)
        label[cat] = 1

        vector = [float(x) for x in parts[1:]]

        labels.append(label)
        data.append(vector)


    labels = np.array(labels)
    data = np.array(data)

    print(labels.shape)
    print(data.shape)

    return data, labels

def update_checkpoint(model, data_idx):
    dir = os.path.join("models", "%s" % (MODEL_BASENAME))

    checkpoint_data_fname = os.path.join(dir, "num_data_read.txt")
    with open(checkpoint_data_fname, "w") as f:
        f.write("%d\n" % (data_idx))


    checkpoint_dir = os.path.join(dir, "checkpoint_model")
    model.save(checkpoint_dir)


def save_best_model(model_q):
    while True:
        model = model_q.get()
        if model is None:
            return
        dir = os.path.join("models", "%s" % (MODEL_BASENAME))
        dir = os.path.join(dir, "best_model")
        model.save(dir)

def get_batch(data, labels, batch_size, data_idx):
    data_idx = data_idx % data.shape[0]
    last_idx = min(data_idx + batch_size, data.shape[0])

    batch_data = data[data_idx:last_idx, :]
    batch_labels = labels[data_idx:last_idx]

    return batch_data, batch_labels


def train_model(model, train_data, train_labels, test_data, test_labels, batch_size = 2048, data_idx = 0, pct_test = .5, init_best_acc = 0.0):

    best_model_q = Queue()
    save_Best_model_thread = threading.Thread(target=save_best_model, args=(best_model_q,))
    save_Best_model_thread.start()

    best_acc = init_best_acc

    print("Starting training with batch size %d" % (batch_size))

    try:
        best_acc = init_best_acc
        idx = 0
        while True:
            
            batch_data, batch_labels = get_batch(train_data, train_labels, batch_size, data_idx)

            model.fit(batch_data, batch_labels, verbose=3)

            test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=0, batch_size=batch_size)

            # Call this to free up memory
            gc.collect()
            keras.backend.clear_session()

            # Update how much data we've seen
            data_idx += batch_size

            # Save the model accordingly and update stats
            if best_acc < test_acc:
                print("Found better accuracy: %f" % (test_acc))
                best_acc = test_acc
                best_clone = tf.keras.models.clone_model(model)
                best_model_q.put(best_clone)
                
            fname = os.path.join("models", "%s" % (MODEL_BASENAME))
            fname = os.path.join(fname, "accuracies.txt")
            with open(fname, "a") as f:
                f.write("%f\n" % (test_acc))

            if idx % 50 == 0:
                update_checkpoint(model, data_idx)
            idx += 1

    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting...")
    except Exception as e:
        print("Encounted unexpected exception...")
        update_checkpoint(model, data_idx)
        raise e

    print("Updating checkpoint")
    best_model_q.put(None)
    update_checkpoint(model, data_idx)

    print("Attempting join...")
    save_Best_model_thread.join()

def get_model():
    inputs = layers.Input(INPUT_VECTOR_SHAPE)
    x = layers.Dense(1024, activation='relu')(inputs)
    x = layers.Concatenate(axis=1)([inputs, x])
    y = layers.Dense(100, activation='softmax')(x)

    model = keras.Model(inputs, y)

    model.summary()
    model.compile(optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_csv', help='CSV with filepaths and labels')
    parser.add_argument('test_csv', help='CSV with filepaths and labels')
    parser.add_argument('-b' '--batch_size', help='Batch size for training', default=32, type=int)

    args = parser.parse_args()

    train_data, train_labels = read_embeddings_csv(args.train_csv)
    test_data, test_labels = read_embeddings_csv(args.test_csv)

    # test_data = read_embeddings_csv(args.test_csv)

    if os.path.exists("models/%s/checkpoint_model" % (MODEL_BASENAME)):
        model = tf.keras.models.load_model("models/%s/checkpoint_model" % (MODEL_BASENAME))
        with open("models/%s/num_data_read.txt" % MODEL_BASENAME, "r") as f:
            data_idx = int(f.read())
            print("Found checkpoint. Resuming training from data index %d" % (data_idx))

        with open("models/transfer/accuracies.txt", "r") as f:
            lines = f.readlines()
            lines = [float(l) for l in lines]
            best_acc = max(lines)
    else:    
        model = get_model()
        data_idx = 0
        best_acc = 0

    if not (os.path.exists("models/%s" % (MODEL_BASENAME))):
        os.mkdir("models/%s" % (MODEL_BASENAME))

    train_model(model, train_data, train_labels, test_data, test_labels, batch_size=args.b__batch_size, data_idx=data_idx, init_best_acc=best_acc)

if __name__ == '__main__':
    main()