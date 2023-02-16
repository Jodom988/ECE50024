from exercise1 import load_train_data, load_test_data, normalize_data

import numpy as np
import matplotlib.pyplot as plt

def part_a_to_b():
    data = load_train_data()
    normalize_data(data)

    X = np.array([data['stature_normalized'], data['bmi_normalized'], [1 for _ in range(data.shape[0])]])
    X = np.matrix.transpose(X)

    y = np.matrix.transpose(np.array(data['class']))

    paren = np.matmul(np.matrix.transpose(X), X)
    weights = np.matmul(np.matmul(np.linalg.inv(paren), np.matrix.transpose(X)), y)

    print(weights)


    slope = weights[0] / -weights[1]
    intercept = weights[2] / -weights[1]
    xs = np.linspace(1.3, 2.0, 100)
    ys = slope * xs + intercept

    plt.scatter(data[data['gender'] == 'female']['stature_normalized'], data[data['gender'] == 'female']['bmi_normalized'], marker='.', color='red')
    plt.scatter(data[data['gender'] == 'male']['stature_normalized'], data[data['gender'] == 'male']['bmi_normalized'], marker='o', edgecolor='blue', facecolor='none')
    plt.plot(xs, ys, '--r')
    plt.xlabel("Stature (normalized)")
    plt.ylabel("BMI (normalized)")
    plt.legend(["Female", "Male", "Decision Boundary"])
    plt.xlim(1.3, 2.0)
    plt.ylim(1, 9)
    # plt.show()


    test_data = load_train_data()
    normalize_data(test_data)

    xs = np.array([np.array(test_data['stature_normalized']), np.array(test_data['bmi_normalized']), np.array([1 for _ in range(data.shape[0])]) ])
    xs = np.matrix.transpose(xs)
    predictions = np.matmul(xs, weights)
    test_data['predictions'] = predictions

    class_predictions = []
    for prediction in predictions:
        if prediction > 0:
            class_predictions.append('male')
        else:
            class_predictions.append('female')
    test_data['class_predictions'] = np.array(class_predictions)

            
    # Male Type 1 error: Pct of samples that should be female but were predicted male (false positives of male)
    fp = test_data[test_data['class_predictions'] == 'male']
    fp = fp[fp['gender'] == 'female']
    print("Type 1: %.2f%%" % (100 * fp.shape[0] / test_data.shape[0]))

    # Male Type 2 error: Pct of samples that should be male but were predicted as female
    fn = test_data[test_data['class_predictions'] == 'female']
    fn = fn[fn['gender'] == 'male']
    print("Type 2: %.2f%%" % (100 * fn.shape[0] / test_data.shape[0]))

    # Precision: true positives / (true positives + false positives)
    tp = test_data[test_data['class_predictions'] == 'male']
    tp = tp[tp['gender'] == 'male']
    print("Precision: %.2f%%" % (100 * tp.shape[0] / (tp.shape[0] + fp.shape[0])))

    # Recall: true positives / (true positives + false negatives)
    print("Precision: %.2f%%" % (100 * tp.shape[0] / (tp.shape[0] + fn.shape[0])))

def main():
    part_a_to_b()
    pass

if __name__ == '__main__':
    main()