import os
import argparse
from tqdm import tqdm

from deepface import DeepFace
from common import read_label_csv
import tensorflow as tf

MODEL = "VGG-Face"

checkpoint_fname = "deepface_embeddings_checkpoint.txt"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label_csv', type=str, help="CSV file containing labels", dest="label_csv")
    parser.add_argument('img_dir', type=str, help="Directory or CSV of images to be evaluated")
    parser.add_argument('out_csv', type=str, help="CSV file to store embeddings")

    args = parser.parse_args()

    img_dir = args.img_dir
    out_csv = args.out_csv

    if args.label_csv is not None:
        imgs = read_label_csv(args.label_csv, img_dir)
    else:
        imgs = [(os.path.join(img_dir, fname), None, None) for fname in os.listdir(img_dir)]

    last_idx_written = -1
    with open(checkpoint_fname, 'r') as f:
        last_idx_written = int(f.read().strip())
    print("Starting from index %d" % (last_idx_written + 1))

    idx = 0
    for img in tqdm(imgs):
        if idx <= last_idx_written:
            idx += 1
            continue

        img_path = img[0]
        label = img[1]

        embedding_obj = None
        try:
            embedding_obj = DeepFace.represent(img_path, model_name=MODEL)
        except ValueError:
            continue
        #embedding_objs_2 = DeepFace.represent(img_path, model_name="Facenet512")

        if len(embedding_obj) == 1:
            vector = embedding_obj[0]['embedding']
            # vector_2 = embedding_objs_2[0]['embedding']
            # vector = vector_1 + vector_2
            with open(out_csv, 'a') as f:
                if label is not None:
                    f.write("%d," % label)

                f.write(",".join([str(x) for x in vector]) + "\n")

        with open(checkpoint_fname, 'w') as f:
            f.write(str(idx))

        idx += 1

if __name__ == '__main__':
    main()