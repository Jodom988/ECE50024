import argparse
import os

from tqdm import tqdm
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import cv2

from common import extract_best_face, extract_all_faces, pil_to_cv2, cv2_to_pil, label_name_mapping

IM_SIZE = (224, 224, 3)

def load_normalize_img(path, policy):
    if (os.path.exists(path) == False):
        print("ERROR: Path does not exist: " + path)
        return None

    original_img = Image.open(path)
    img = pil_to_cv2(original_img)

    face = None
    none_found = False
    if policy == 'multiple':
        face = extract_all_faces(img)
    else:
        face = extract_best_face(img, policy=policy)
    
    if face is None:
        # Could not extract face, just use original image
        none_foune = True
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
        return ([img], none_found)
    
    imgs = None
    if policy == 'multiple':
        imgs = [cv2_to_pil(f) for f in face]
        imgs = [img.resize((IM_SIZE[0], IM_SIZE[1])) for img in imgs]
    else:
        img = cv2_to_pil(face)
        img = img.resize((IM_SIZE[0], IM_SIZE[1]))

        imgs = [img]

    # if face is not None:
    #     cv2.imshow("img", imgs)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # save image
    # img.save("tmp.png")

    # if len(imgs) > 1:
    #     print("Found %d faces" % len(imgs))
    #     original_img.save("tmp.png")
    #     input()
    #     for img in imgs:
    #         img.save("tmp.png")
    #         input()
    

    return (imgs, none_found)

def load_imgs(img_dir, policy):
    data = list()
    files = os.listdir(img_dir)
    none_found_ct = 0
    for img_name in tqdm(files, desc="Loading images"):
        img_path = os.path.join(img_dir, img_name)
        # print(img_path)

        imgs, none_found = load_normalize_img(img_path, policy)

        if none_found:
            none_found_ct += 1
        imgs = [np.array(img) for img in imgs]
            
        img_name = os.path.splitext(img_name)[0]
        data.append((imgs, img_name))

    print("None found for %d, (%.2f)" % (none_found_ct, none_found_ct / len(files)))

    return data

def make_predictions_mult(faces, model):

    img_names = list()
    names = list()
    label_to_name = label_name_mapping()

    for i in tqdm(range(len(faces)), desc="Making predictions"):
        imgs, img_name = faces[i]

        imgs = np.array(imgs)
        predictions = model.predict(imgs, verbose=3)

        class_predictions = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1)
        best_idx = np.argmax(confidence)
        class_prediction = class_predictions[best_idx]

        img_names.append(int(img_name))
        names.append(label_to_name[class_prediction])

    return img_names, names


def make_predictions_single(faces, model):

    imgs = [img[0] for img, img_name in faces]
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

    return img_names, names


def write_predictions(img_names, names, out_csv):
    # sort img_names array
    for i in range(len(img_names)):
        for j in range(len(img_names)):
            if img_names[i] < img_names[j]:
                img_names[i], img_names[j] = img_names[j], img_names[i]
                names[i], names[j] = names[j], names[i]
    
    # Write to file
    with open(out_csv, 'w') as f:
        f.write("Id,Category\n")
        for i in range(len(img_names)):
            f.write("%d,%s\n" % (img_names[i], names[i]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', help='Directory with images')
    parser.add_argument('model_dir', help='Directory with model')
    parser.add_argument('out_csv', help='CSV File containing predicted labels')
    parser.add_argument('policy', help='Policy for face extraction (multiple, closest_to_center, largest, single)')
    args = parser.parse_args()

    if args.policy not in ['multiple', 'closest_to_center', 'largest', 'single']:
        print("ERROR: Invalid policy")
        return
    
    faces = load_imgs(args.img_dir, args.policy)

    model = tf.keras.models.load_model(args.model_dir)
    model.summary()

    if args.policy == 'multiple':
        img_name, name_precition = make_predictions_mult(faces, model)
    else:
        img_name, name_precition = make_predictions_single(faces, model)

    write_predictions(img_name, name_precition, args.out_csv)


if __name__ == "__main__":
    main()