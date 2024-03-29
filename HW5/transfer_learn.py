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
from tensorflow.keras import layers
from keras_vggface.vggface import VGGFace

from keras_facenet import FaceNet


IM_SIZE = (224, 224, 3)
MODEL_BASENAME = "transfer"
N_LABELS = 100

def load_batch(data, idxes):
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
        label = np.zeros(N_LABELS)
        label[class_label] = 1
        labels.append(label)
        # labels.append(class_label)

    imgs = np.array(imgs)
    labels = np.array(labels)

    return imgs, labels

def load_normalize_img(path):
    if (os.path.exists(path) == False):
        print("ERROR: Path does not exist: " + path)
        return None

    img = Image.open(path)
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

def load_imgs_into_new_img_queue(q: Queue, data, stop_event, start_idx = 0):
    curr_idx = start_idx
    while True:
        batch_len = min(64, len(data) - curr_idx)
        idxes = range(curr_idx, curr_idx + batch_len)
        imgs, labels = load_batch(data, idxes)
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

        imgs, labels = load_batch(data, idxes)
        
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


def save_best_model(model_q):
    while True:
        model = model_q.get()
        if model is None:
            return
        dir = os.path.join("models", "%s" % (MODEL_BASENAME))
        dir = os.path.join(dir, "best_model")
        model.save(dir)



def train_model(model, train_data, test_data, batch_size = 64, num_batches = 25, data_idx = 0, pct_test = .5, init_best_acc = 0.0):

    best_model_q = Queue()
    save_Best_model_thread = threading.Thread(target=save_best_model, args=(best_model_q,))
    save_Best_model_thread.start()

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
    test_imgs, test_labels = load_batch(test_data, range(len(test_data)))
    test_imgs = np.array(test_imgs)
    test_labels = np.array(test_labels)
    
    print("Starting training with batch size %d and %d number of batches" % (batch_size, num_batches))
    if num_batches == -1:
        iterator = range(10000000000000)   # Pseudo Forever
    else:
        iterator = tqdm(range(num_batches))

    try:
        best_acc = init_best_acc
        for epoch in iterator:
            # Load new images to train the model on
            train_imgs, train_labels = load_imgs_from_queue(new_img_q, batch_size)
            train_imgs = np.array(train_imgs)
            train_labels = np.array(train_labels)

            # Load old images to train the model on to avoid catestrophic forgetting
            train_imgs_old, train_labels_old = load_imgs_from_queue(old_img_q, batch_size)
            train_imgs_old = np.array(train_imgs_old)
            train_labels_old = np.array(train_labels_old)

            train_imgs = np.vstack([train_imgs, train_imgs_old])
            train_labels = np.vstack([train_labels, train_labels_old])

            model.fit(train_imgs, train_labels, verbose=3)

            # Evaluate the model
            if pct_test < 1 and pct_test > 0:
                test_idexes = random.sample(range(len(test_data)), int(len(test_data) * pct_test))
                curr_test_imgs = test_imgs[test_idexes]
                curr_test_labels = test_labels[test_idexes]
            else:
                curr_test_imgs = test_imgs
                curr_test_labels = test_labels
            test_loss, test_acc = model.evaluate(curr_test_imgs,  curr_test_labels, verbose=0, batch_size=batch_size)

            # Call this to free up memory
            gc.collect()
            keras.backend.clear_session()

            # Update how much data we've seen
            data_idx += batch_size
            last_idx[0] = min(data_idx, len(train_data))

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

            if epoch % 50 == 0:
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
    best_model_q.put(None)

    # Close the new image getter thread
    while True:
        print("Attempting join... (1/3)")
        for _ in range(new_img_q.qsize()):
            new_img_q.get(timeout=.1)
        new_imgs_load_thread.join(timeout=.5)
        if not new_imgs_load_thread.is_alive():
            break
    
    # Close the old image getter thread
    while True:
        print("Attempting join... (2/3)")
        for _ in range(old_img_q.qsize()):
            old_img_q.get(timeout=.1)
        old_imgs_load_thread.join(timeout=.5)
        if not old_imgs_load_thread.is_alive():
            break

    print("Attempting join... (3/3)")
    save_Best_model_thread.join()

def get_model_facenet():
    print("Reading facenet_keras.h5...")
    facenet_base = keras.models.load_model('models/facenet_keras.h5')

    facenet_base.summary()

def get_model_vgg():
    vggface_base = VGGFace(model='resnet50', include_top=False, input_shape=IM_SIZE)
    vggface_base.trainable = False

    last_layer = vggface_base.get_layer('avg_pool').output

    # # Build the model
    inputs = tf.keras.Input(shape=IM_SIZE)
    x = vggface_base(inputs)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(2048, name='hidden1', activation='relu')(x)
    x = layers.Dense(1024, name='hidden2', activation='relu')(x)
    out = layers.Dense(N_LABELS, name='Classifier', activation='softmax')(x)
    model = keras.Model(inputs, out)
    
    model.summary()

    model.compile(optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_csv', help='CSV with filepaths and labels')
    parser.add_argument('test_csv', help='CSV with filepaths and labels')
    parser.add_argument('train_img_dir', help='Directory with images')
    parser.add_argument('test_img_dir', help='Directory with images')
    parser.add_argument('-b' '--batch_size', help='Batch size for training', default=1024, type=int)
    parser.add_argument('-n' '--num_batches', help='Number of ephocs', default=25, type=int)
    parser.add_argument('-p' '--pct_test_data', help='Percent of test data to use', default=None, type=int)

    args = parser.parse_args()

    train_data = read_label_csv(args.train_csv, args.train_img_dir)
    test_data = read_label_csv(args.test_csv, args.test_img_dir)

    if args.p__pct_test_data:
        print("TRIMMING TEST DATA")
        test_data = test_data[:int(len(test_data) * args.p__pct_test_data / 100)]

    if os.path.exists("models/transfer/checkpoint_model"):
        model = tf.keras.models.load_model("models/transfer/checkpoint_model")
        model.trainable = True
        with open("models/transfer/num_data_read.txt", "r") as f:
            data_idx = int(f.read())
            print("Found checkpoint. Resuming training from data index %d" % (data_idx))

        with open("models/transfer/accuracies.txt", "r") as f:
            lines = f.readlines()
            lines = [float(l) for l in lines]
            best_acc = max(lines)
    else:    
        model = get_model_vgg()
        data_idx = 0
        best_acc = 0

    if not (os.path.exists("models/transfer")):
        os.mkdir("models/transfer")

    train_model(model, train_data, test_data, batch_size=args.b__batch_size, num_batches=args.n__num_batches, data_idx=data_idx, init_best_acc=best_acc)

if __name__ == '__main__':
    main()