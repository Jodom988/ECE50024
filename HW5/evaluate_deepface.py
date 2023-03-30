import argparse
import os

from common import read_label_csv, label_name_mapping

from deepface import DeepFace
from tqdm import tqdm
import numpy as np
import cv2

MODEL = "Facenet512"

def get_identity_from_path(path):
    identity = path.split("/")[-1].split(".")[0]
    identity = ''.join([i for i in identity if not i.isdigit()])
    return identity

def get_identity_distance_dict(dfs):
    name_dists = dict()
    for df in dfs:
        for i in range(len(df)):
            ident = get_identity_from_path(df['identity'][i])
            # dist = df['VGG-Face_euclidean_l2'][i]
            dist = df['Facenet512_euclidean_l2'][i]
            
            if ident in name_dists:
                name_dists[ident].append(dist)
            else:
                name_dists[ident] = [dist]

    return name_dists

def find_best(name_dists):
    smallest_dist = 10000000000000
    smallest_dist_name = None
    most_entries = 0
    most_entries_name = None

    for ident in name_dists:

        if len(name_dists[ident]) > most_entries:
            most_entries = len(name_dists[ident])
            most_entries_name = ident

        name_dists[ident] = np.array(name_dists[ident])
        mean_dist = np.mean(name_dists[ident])
        if mean_dist < smallest_dist:
            smallest_dist = mean_dist
            smallest_dist_name = ident

    return smallest_dist_name, most_entries_name

def predict(db_dir, img_dir, out_csv, dist_csv = None):
    # check if path extists
    if os.path.exists(out_csv):
        with open(out_csv, "r") as f:
            lines = f.readlines()
        ids_done = [int(line.split(",")[0]) for line in lines]
    else:
        ids_done = []

    fnames = os.listdir(img_dir)
    # fnames = ['138.jpg', '206.jpg', '240.jpg', '282.jpg', '293.jpg']

    print("== Starting Predictions ==")
    for i in tqdm(range(len(fnames))):
        fname = fnames[i]
        img_id = int(fname.split(".")[0])

        img_path = os.path.join(img_dir, fname)

        if img_id in ids_done:
            continue

        try:
            dfs = DeepFace.find(img_path=img_path, db_path=db_dir, enforce_detection=False, distance_metric="euclidean_l2", model_name=MODEL)
        except KeyboardInterrupt:
            print("Keyboard interrupt, exiting...")
            break
        except Exception as e:
            ids_done.append(img_id)
            print("===================================================")
            print("Error with image %s" % img_path)
            print("===================================================")
            continue

        
        name_dists = get_identity_distance_dict(dfs)
        best_by_dist, best_by_count = find_best(name_dists)
        ids_done.append(img_id)

        if best_by_count is None:
            best_by_count = "=====UNKNOWN====="

        with open(out_csv, "a") as f:
            name = best_by_count.replace("_", " ")
            f.write("%d,%s\n" % (img_id, name))
            print("Writing %d,%s" % (img_id, name))



def evaluate(db_dir, img_dir, label_csv, dist_csv = None):
    test_data = read_label_csv(label_csv, img_dir)



    correct_count_method = 0
    correct_avg_method = 0
    count = 0
    for i in tqdm(range(len(test_data))):
        img_path, label, gt_name = test_data[i]
        gt_name = gt_name.replace(" ", "_") # Ground truth name
        img_id = int(img_path.split("/")[-1].split(".")[0])

        try:
            dfs = DeepFace.find(img_path=img_path, db_path=db_dir, enforce_detection=False, distance_metric="euclidean_l2", model_name=MODEL)
        except KeyboardInterrupt:
            print("Keyboard interrupt, exiting...")
            break
        except Exception as e:
            print("Error with image %s" % img_path)
            continue

        name_dists = get_identity_distance_dict(dfs)
        best_by_dist, best_by_count = find_best(name_dists)
            
        if best_by_dist == gt_name:
            correct_avg_method += 1
        if best_by_count == gt_name:
            correct_count_method += 1

        count += 1

        print("=====================================")
        print("Smallest dist: %s" % best_by_dist)
        print("Most entries: %s" % best_by_count)    
        print("Actual: %s" % gt_name) 
        print("Correct by avg: %.2f" % (correct_avg_method / count))
        print("Correct by count: %.2f" % (correct_count_method / count))
        print("%.2f%% complete" % (count / len(test_data) * 100))
        print("=====================================")

        if dist_csv is not None:
            with open(dist_csv, "a") as f:
                for name in name_dists:
                    dists = name_dists[name]
                    dist_str = ",".join([str(dist) for dist in dists])
                    is_right = 1 if name == gt_name else 0
                    f.write("%d,%d,%s\n" % (img_id, is_right, dist_str))

def sort(csv_file):
    with open(csv_file, "r") as f:
        lines = f.readlines()

    lines = [(int(line.split(",")[0]), line.split(",")[1][:-1]) for line in lines]
    lines_sorted = sorted(lines, key=lambda x: x[0])

    [print(line) for line in lines_sorted]

    with open(csv_file, "w") as f:
        for line in lines_sorted:
            f.write("%d,%s\n" % (line[0], line[1]))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_dir", type=str)
    parser.add_argument("img_dir", type=str, help="Directory of images to be evaluated")

    parser.add_argument("-l", "--labels", type=str, help="CSV holding labels of images to be evaluated", dest="label_csv")
    parser.add_argument("-o", "--output", type=str, help="Output file to write results to", dest="output_file")
    parser.add_argument("-s", "--sort_csv", type=str, help="CSV file to sort by", dest="sort_csv")
    parser.add_argument("-d", "--distances_csv", type=str, help="CSV file to store distances", dest="dist_csv")

    args = parser.parse_args()

    none_ct = 0
    if args.output_file is not None:
        none_ct += 1
    if args.label_csv is not None:
        none_ct += 1
    if args.sort_csv is not None:
        none_ct += 1

    if none_ct > 1:
        print("Error! Must provide only one of --labels, --output, or --sort_csv")
        return
    elif none_ct == 0:
        print("Error! Must provide one of --labels, --output, or --sort_csv")
        return
    
    if args.label_csv is not None:
        evaluate(args.db_dir, args.img_dir, args.label_csv, dist_csv = args.dist_csv)
    elif args.output_file is not None:
        predict(args.db_dir, args.img_dir, args.output_file, dist_csv = args.dist_csv)
    elif args.sort_csv is not None:
        sort(args.sort_csv)
    

if __name__ == "__main__":
    main()