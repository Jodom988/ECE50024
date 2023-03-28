import os
import argparse
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm

from common import pil_to_cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", help="Directory of images to be cleaned")
    parser.add_argument("out_dir", help="Directory to write cleaned images to")

    args = parser.parse_args()
    
    if os.path.exists(args.out_dir):
        raise Exception("Output directory already exists")
    
    os.mkdir(args.out_dir)

    files = os.listdir(args.in_dir)
    for i in tqdm(range(len(files))):
        try:
            file = files[i]
            img = Image.open(os.path.join(args.in_dir, file))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img.save(os.path.join(args.out_dir, file))
        except Exception as e:
            print("Error with file %s" % file)
            raise(e)


if __name__ == "__main__":
    main()
