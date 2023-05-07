import os
import gc

from tqdm import tqdm
from deepface import DeepFace
import cv2
import pandas as pd

import tensorflow as tf

def main():
    img_list_fpath = '../../data/Project/list_attr_celeba.txt'
    out_labels_csv = '../../data/Project/celeba_cropped_labels.csv'
    out_embeddings_csv = '../../data/Project/celeba_cropped_embeddings.csv'
    in_img_dir = '../../../ECE634/final-project-team9-joshmajors/data/celeba_aligned_imgs/'
    out_img_dir = '../../data/Project/celebA_cropped'

    df = pd.read_csv(img_list_fpath, sep='\s+')
    print(df)

    last_done = -1
    if os.path.exists(out_labels_csv):
        already_written = pd.read_csv(out_labels_csv)
        already_written = already_written['fname'].values
        last_done = int(already_written[-1].split('.')[0])

    else:
        with open(out_labels_csv, 'w') as f:
            f.write(','.join(list(df.columns)))
            f.write('\n')

        with open(out_embeddings_csv, 'w') as f:
            pass

    for i in tqdm(range(len(df))):
        if i <= last_done:
            continue

        row = df.iloc[i]

        fname = None
        vals_str = ''
        for item in row.items():
            if item[0] == 'fname':
                fname = item[1]
            else:
                if item[1] == 1:
                    vals_str += ',1'
                elif item[1] == -1:
                    vals_str += ',0'
                else:
                    print("UNEXPECTED VALUE: ", item[1])
                    raise Exception("Unexpected value")
            
        in_fpath = os.path.join(in_img_dir, fname)

        embedding_objs = None
        try:
            embedding_objs = DeepFace.represent(in_fpath, model_name='Facenet')
        except ValueError:
            pass

        
        if embedding_objs is None:
            # print("No face detected")
            continue
        elif len(embedding_objs) > 1:
            # print("More than one face detected")
            continue

        with open(out_labels_csv, 'a') as f:
            f.write(fname + vals_str + '\n')

        with open(out_embeddings_csv, 'a') as f:
            f.write(fname + ',' + ','.join([str(x) for x in embedding_objs[0]['embedding']]) + '\n')

        img = cv2.imread(in_fpath)
        
        embedding_obj = embedding_objs[0]
        x = embedding_obj['facial_area']['x']
        y = embedding_obj['facial_area']['y']
        w = embedding_obj['facial_area']['w']
        h = embedding_obj['facial_area']['h']

        img = img[y:y+h, x:x+w]

        out_fpath = os.path.join(out_img_dir, fname)
        cv2.imwrite(out_fpath, img)

        gc.collect()
        tf.keras.backend.clear_session()


    pass

if __name__ == '__main__':
    main()