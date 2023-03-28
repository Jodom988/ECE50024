import os
import shutil

from common import read_label_csv

from tqdm import tqdm

def main():
    img_data = read_label_csv("../data/HW5/train_train_cropped_single_80_20.csv", "../data/HW5/train_cropped_single")
    img_db_dir = "../data/HW5/train_train_db_copped_single_all"

    seq_idx_by_name = dict()

    for i in tqdm(range(len(img_data))):
        data = img_data[i]
        img_path, _, name = data
        name = name.replace(" ", "_")
        tmp_dir = os.path.join(img_db_dir, name)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        seq_num = 1
        if name in seq_idx_by_name:
            seq_num = seq_idx_by_name[name]
            # if seq_num > 200:
            #     continue
            seq_idx_by_name[name] += 1
        else:
            seq_idx_by_name[name] = 1

        inpath = img_path
        outpath = os.path.join(img_db_dir, name)
        outpath = os.path.join(outpath, "%s%d.jpg" % (name, seq_num))

        shutil.copyfile(inpath, outpath)


    pass

if __name__ == "__main__":
    main()