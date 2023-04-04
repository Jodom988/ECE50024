import os
import argparse
from tqdm import tqdm
import cv2

from deepface import DeepFace
from common import read_label_csv

MODEL = "VGG-Face"

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

    found_multiple = 0
    found_none = 0
    for img in tqdm(imgs):
        img_path = img[0]
        label = img[1]

        embedding_objs = DeepFace.represent(img_path, model_name=MODEL, enforce_detection=False)

        if (len(embedding_objs) == 1):
            vector = embedding_objs[0]['embedding']
            with open(out_csv, 'a') as f:
                if label is not None:
                    f.write("%d," % label)

                f.write(",".join([str(x) for x in vector]) + "\n")
        elif (len(embedding_objs) >= 1):
            found_multiple += 1
        elif (len(embedding_objs) == 0):
            found_none += 1
        else:
            print("ERROR: Invalid number of embeddings")

    print("Found multiple: %d (%.2f)" % (found_multiple, found_multiple / len(imgs)))
    print("Found none: %d (%.2f)" % (found_none, found_none / len(imgs)))
    print("Total: %d" % len(imgs))



if __name__ == '__main__':
    main()