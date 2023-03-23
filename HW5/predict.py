import argparse
import os

from tqdm import tqdm
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import cv2

from common import extract_face, pil_to_cv2, cv2_to_pil, label_name_mapping

IM_SIZE = (224, 224, 3)

def load_normalize_img(path):
    if (os.path.exists(path) == False):
        print("ERROR: Path does not exist: " + path)
        return None

    img = Image.open(path)

    img = pil_to_cv2(img)
    face = extract_face(img)
    if face is not None:
        img = cv2_to_pil(face)
        img = img.resize((IM_SIZE[0], IM_SIZE[1]))
    elif face is None:
        # Could not extract face, just use original image
        img = cv2_to_pil(img)
        width, height = img.size
        if width < height:
            new_height = IM_SIZE[1]
            new_width = int(width * (new_height / height))
            img = img.resize((new_width, new_height))
        else:
            new_width = IM_SIZE[0]
            new_height = int(height * (new_width / width))
            img = img.resize((new_width, new_height))

        h_border = (IM_SIZE[0] - new_width) // 2
        v_border = (IM_SIZE[1] - new_height) // 2
        img = ImageOps.expand(img, border=(h_border, v_border, h_border, v_border), fill=(0, 0, 0))
        img = img.resize((IM_SIZE[0], IM_SIZE[1]))

    # if face is None:
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # save image
    # img.save("tmp.png")

    return img

def load_imgs(img_dir):
    imgs = list()
    for img_name in tqdm(os.listdir(img_dir), desc="Loading images"):
        img_path = os.path.join(img_dir, img_name)
        # print(img_path)

        img = load_normalize_img(img_path)
        img = np.array(img)
            
        img_name = os.path.splitext(img_name)[0]
        imgs.append((img, img_name))

    return imgs

def make_predictions(faces, model, out_csv):

    imgs = [img for img, img_name in faces]
    img_names = [int(img_name) for img, img_name in faces]

    if (len(imgs) != len(img_names)):
        print("ERROR: Length of imgs and img_names do not match")
        return

    imgs = np.array(imgs)

    class_predictions = model.predict(imgs)

    class_predictions = np.argmax(class_predictions, axis=1)

    label_to_name = label_name_mapping()
    names = list()
    for i in range(len(class_predictions)):
        names.append(label_to_name[class_predictions[i]])

    # sort img_names array
    for i in range(len(img_names)):
        for j in range(len(img_names)):
            if img_names[i] < img_names[j]:
                img_names[i], img_names[j] = img_names[j], img_names[i]
                names[i], names[j] = names[j], names[i]
    
    with open(out_csv, 'w') as f:
        f.write("Id,Category\n")
        for i in range(len(img_names)):
            f.write("%d,%s\n" % (img_names[i], names[i]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', help='Directory with images')
    parser.add_argument('model_dir', help='Directory with model')
    parser.add_argument('out_csv', help='CSV File containing predicted labels')
    args = parser.parse_args()

    faces = load_imgs(args.img_dir)

    model = tf.keras.models.load_model(args.model_dir)
    model.summary()

    make_predictions(faces, model, args.out_csv)


if __name__ == "__main__":
    main()