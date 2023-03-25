import os

import cv2
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np
import dlib

def label_name_mapping(fname = '../data/HW5/category.csv'):
    with open(fname, 'r') as f:
        lines = f.readlines()

    label_to_name = dict()
    for i, lin in enumerate(lines):
        if i == 0:
            continue
        idx, name = lin.split(',')
        idx = int(idx)
        label_to_name[idx] = name.strip()

    return label_to_name

def name_label_mapping(fname = '../data/HW5/category.csv'):
    with open(fname, 'r') as f:
        lines = f.readlines()

    name_to_label = dict()
    for i, lin in enumerate(lines):
        if i == 0:
            continue
        idx, name = lin.split(',')
        idx = int(idx)
        name_to_label[name.strip()] = idx

    return name_to_label

def read_label_csv(fname, img_dir):
    with open(fname, 'r') as f:
        lines = f.readlines()

    name_to_label = name_label_mapping()

    data = []
    for i, lin in enumerate(lines):
        if i == 0:
            continue
        idx, path, name = lin.split(',')
        path = os.path.join(img_dir, path)
        name = name.strip()
        label = name_to_label[name]        
        data.append((path, label, name))

    return data

def getResultCenter(result):
    x1, y1, w, h = result
    x2, y2 = x1 + w, y1 + h
    return ((x1+x2)/2, (y1+y2)/2)

def getDistToCenter(p1, img):
    center = (img.shape[1]/2, img.shape[0]/2)
    return getDistance(p1, center)

def getDistance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def pil_to_cv2(pil_img):
    open_cv_image = np.array(pil_img)
    if len(open_cv_image.shape) == 2:
        open_cv_image2d = open_cv_image
        open_cv_image = np.zeros((open_cv_image2d.shape[0], open_cv_image2d.shape[1], 3), dtype=np.uint8)
        open_cv_image[:, :, 0] = open_cv_image2d
        open_cv_image[:, :, 1] = open_cv_image2d
        open_cv_image[:, :, 2] = open_cv_image2d
    elif len(open_cv_image.shape) == 3:
        open_cv_image = open_cv_image[:, :, ::-1].copy()
    else:
        raise Exception("Invalid image shape: {}".format(open_cv_image.shape))
    return open_cv_image

def cv2_to_pil(cv2_img):
    color_converted = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_converted)
    return pil_image

CLASSIFIER = cv2.CascadeClassifier('models/haar_cascade/haarcascade_frontalface2.xml')  # Has about 20% cases where it fails to detect a face
DETECTOR = dlib.get_frontal_face_detector()     # Has about 11% cases where it fails to detect a face

# Expects an openCV image and returns the bounding box of the face (x1, y1, w, h)
def get_face_bounds(img, extend=0.50):
    #faces = CLASSIFIER.detectMultiScale(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = DETECTOR(gray, 1) # results

    closest_dist = None
    best_result = None
    for result in faces:
        x1 = result.left()
        x2 = result.right()
        y1 = result.top()
        y2 = result.bottom()
        w = x2 - x1
        h = y2 - y1
        x1 += int(-extend * w)
        x2 += int(extend * w)
        y1 += int(-extend * h)
        y2 += int(extend * h)

        result = (x1, y1, x2-x1, y2-y1)
        curr_center = getResultCenter(result)
        
        dist = getDistToCenter(curr_center, img)
        if best_result is None or dist < closest_dist:
            closest_dist = dist
            best_result = result

    return best_result

# Expects an openCV image and returns a new image with the face cropped and resized to 128x128, returns None if no face found
def extract_face(img):
    bounding_box = get_face_bounds(img)

    if bounding_box is None:
        return None
    
    x1, y1, w, h = bounding_box
    x2, y2 = x1 + w, y1 + h

    if (w > h):
        diff = w - h
        y1 -= diff // 2
        y2 += diff // 2
        if (diff % 2 == 1):
            y2 += 1

        h = y2-y1
    elif (h > w):
        diff = h - w
        x1 -= diff // 2
        x2 += diff // 2
        if (diff % 2 == 1):
            x2 += 1

        w = x2-x1

    diff = 0
    if (x1 < 0) or (y1 < 0) or (x2 > img.shape[1]) or (y2 > img.shape[0]):
        diff = 0
        if x1 < 0:
            diff = abs(x1)
        if y1 < 0 and abs(y1) > diff:
            diff = abs(y1)
        if x2 > img.shape[1] and abs(x2 - img.shape[1]) > diff:
            diff = abs(x2 - img.shape[1])
        if y2 > img.shape[0] and abs(y2 - img.shape[0]) > diff:
            diff = abs(y2 - img.shape[0])
        new_img = cv2.copyMakeBorder(img, diff, diff, diff, diff, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        x1 += diff
        x2 += diff
        y1 += diff
        y2 += diff
        img = new_img

    if (w != h):
        print("ERROR! Not square")
    
    new_img = img[y1:y2, x1:x2]
    new_img = cv2.resize(new_img, (128, 128))

    return new_img