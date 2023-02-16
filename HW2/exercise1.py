import pandas as pd
import matplotlib.pyplot as plt

def load_data(path_male, path_female):
    data = dict()
    male = pd.read_csv(path_male)
    male.rename(columns={"male_bmi": "bmi", "male_stature_mm": "stature"}, inplace=True)
    male['gender'] = 'male'
    male['class'] = 1
    
    female = pd.read_csv(path_female)
    female.rename(columns={"female_bmi": "bmi", "female_stature_mm": "stature"}, inplace=True)
    female['gender'] = 'female'
    female['class'] = -1

    data = pd.concat([male, female], ignore_index=True)

    return data

def load_test_data():
    return load_data('data/male_test_data.csv', 'data/female_test_data.csv')

def load_train_data():
    return load_data('data/male_train_data.csv', 'data/female_train_data.csv')

def normalize_data(data):
    data['stature_normalized'] = data['stature'] / 1000
    data['bmi_normalized'] = data['bmi'] / 10

def main():
    train_data = load_train_data()
    normalize_data(train_data)
    print(train_data[train_data['gender'] == 'male'][:10])
    print("\n")
    print(train_data[train_data['gender'] == 'female'][:10])

    plt.scatter(train_data[train_data['gender'] == 'male']['bmi'], train_data[train_data['gender'] == 'male']['stature'], marker='o')
    plt.scatter(train_data[train_data['gender'] == 'female']['bmi'], train_data[train_data['gender'] == 'female']['stature'], marker='x')
    plt.xlabel("BMI")
    plt.ylabel("Stature")
    plt.legend(["Female", "Male"])
    plt.show()

if __name__ == '__main__':
    main()