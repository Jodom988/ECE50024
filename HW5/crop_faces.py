import argparse
import os 

import cv2
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np

from common import *

def crop_faces(data, out_dir, out_csv, policy):

    none_count = 0
    for i in tqdm(range(len(data)), desc="Cropping Images"):
        path, _, name = data[i]
        pil_img = Image.open(path).convert('RGB')
        open_cv_image = pil_to_cv2(pil_img)
        if open_cv_image is None:
            print("None image")

        face = extract_face(open_cv_image, policy=policy)

        if face is None:
            none_count += 1
            continue

        new_fname = os.path.basename(path)
        new_path = os.path.join(out_dir, new_fname)

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