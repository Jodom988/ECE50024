import random


def split_data(data, train_ratio):
    train_size = int(len(data) * train_ratio)
    random.shuffle(data)
    train_data = data[:train_size]
    test_data = data[train_size:]

    return train_data, test_data

def read_label_csv(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()

    data = []
    for i, lin in enumerate(lines):
        if i == 0:
            continue
        idx, path, name = lin.split(',')
        name = name.strip()
        data.append((idx, path, name))

    return data

def main():
    data = read_label_csv('../data/HW5/train.csv')
    train_data, test_data = split_data(data, 0.8)

    with open("train_train_80_20.csv", "w") as f:
        f.write(",File Name,Category\n")
        for idx, path, name in train_data:
            f.write("{},{},{}\n".format(idx, path, name))

    with open("train_test_80_20.csv", "w") as f:
        f.write(",File Name,Category\n")
        for idx, path, name in test_data:
            f.write("{},{},{}\n".format(idx, path, name))

        
    pass

if __name__ == "__main__":
    main()