import argparse
import os

from common import label_name_mapping

from tqdm import tqdm
from deepface import DeepFace
import tensorflow as tf
import numpy as np
import cv2

MODEL = "VGG-Face"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str, help="Directory or CSV of images to be evaluated")
    parser.add_argument('out_csv', type=str, help="CSV file to store embeddings")
    parser.add_argument('model_dir', type=str, help="Directory with model")

    args = parser.parse_args()

    img_dir = args.img_dir

    img_fnames = [fname for fname in os.listdir(img_dir)]

    model = tf.keras.models.load_model(args.model_dir)
    label_to_name = label_name_mapping()

    for img_fname in tqdm(img_fnames):
        img_path = os.path.join(img_dir, img_fname)
        img_id = int(img_fname.split('.')[0])

        # cv2.imshow("img", cv2.imread(img_path))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        embedding_objs = DeepFace.represent(img_path, model_name=MODEL, enforce_detection=False)

        best_class = -1
        best_prob = -1
        for embedding in embedding_objs:
            embedding = embedding['embedding']
            embedding = np.array(embedding)
            embedding = embedding.reshape((1, embedding.shape[0]))
            prediction = model.predict(embedding, verbose=3)
            prob = np.max(prediction)
            if prob > best_prob:
                best_prob = prob
                best_class = np.argmax(prediction)

        
        name = label_to_name[best_class]
        with open(args.out_csv, 'a') as f:
            f.write("%d,%s\n" % (img_id, name))

    pass

if __name__ == "__main__":
    main()