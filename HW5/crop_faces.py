import argparse
import os 
import random

import cv2
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np

from common import *

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def crop_faces(data, out_dir, out_csv, policy):

    none_count = 0
    id = 0
    for i in tqdm(range(len(data)), desc="Cropping Images"):
        path, _, name = data[i]
        pil_img = Image.open(path).convert('RGB')
        open_cv_image = pil_to_cv2(pil_img)
        if open_cv_image is None:
            print("None image")

        # Get face
        face = extract_best_face(open_cv_image, policy=policy)

        if face is None:
            none_count += 1
            continue

        new_fname = "%d.jpg" % id
        new_path = os.path.join(out_dir, new_fname)
        id += 1

        cv2.imwrite(new_path, face)
        with open(out_csv, 'a') as f:
            fname_no_ext = os.path.splitext(new_fname)[0]
            f.write("%s,%s,%s\n" % (fname_no_ext, new_fname, name))

        # Mutate image and get face again
        mutated_cv_image = open_cv_image.copy()
        mutated_cv_image = cv2.convertScaleAbs(mutated_cv_image, beta=random.randint(-50, 50))
        mutated_cv_image = cv2.flip(mutated_cv_image, 1)
        mutated_cv_image = rotate_image(mutated_cv_image, random.randint(-25, 25))

        face = extract_best_face(mutated_cv_image, policy=policy)

        if face is None:
            continue

        new_fname = "%d.jpg" % id
        new_path = os.path.join(out_dir, new_fname)
        id += 1

        cv2.imwrite(new_path, face)
        with open(out_csv, 'a') as f:
            fname_no_ext = os.path.splitext(new_fname)[0]
            f.write("%s,%s,%s\n" % (fname_no_ext, new_fname, name))

        
    print("Percent with no label: %f" % (none_count / len(data)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_csv', type=str)
    parser.add_argument('img_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('out_csv', type=str)
    parser.add_argument('policy', type=str, help="Policy to use for cropping. Options: 'closest_to_center', 'largest', 'only_one'")
    args = parser.parse_args()

    data = read_label_csv(args.img_csv, args.img_dir)
    crop_faces(data, args.out_dir, args.out_csv, args.policy)
    
        

if __name__ == '__main__':
    main()