import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

def get_likelihood(mu, sigma, sigma_inv, log_sigma, log_pi, x):
    x_minus_mu = x - mu
    x_minus_mu_t = np.transpose(x_minus_mu)

    term1 = -0.5 * np.matmul(np.matmul(x_minus_mu_t, sigma_inv), x_minus_mu)[0, 0]
    term2 = log_pi
    term3 = log_sigma

    return term1 + term2 + term3

def partb_c(predict_img, ground_truth_img):
    train_cat = np.matrix(np.loadtxt('data/train_cat.txt', delimiter=','))
    train_grass = np.matrix(np.loadtxt('data/train_grass.txt', delimiter=','))

    mu_1, sigma_1 = get_mu_sigma(train_cat)
    mu_0, sigma_0 = get_mu_sigma(train_grass)

    sigma_1_inv = np.linalg.inv(sigma_1)
    sigma_0_inv = np.linalg.inv(sigma_0)
    log_sigma_1 = -0.5 * np.log(np.linalg.det(sigma_1))
    log_sigma_0 = -0.5 * np.log(np.linalg.det(sigma_0))

    pi_1 = train_cat.shape[1] / (train_cat.shape[1] + train_grass.shape[1])
    pi_0 = train_grass.shape[1] / (train_cat.shape[1] + train_grass.shape[1])

    log_pi_1 = np.log(pi_1)
    log_pi_0 = np.log(pi_0)

    img = plt.imread(predict_img) / 255
    expected = plt.imread(ground_truth_img) / 255

    M,N = img.shape
    prediction = np.zeros((M-8,N-8))

    detections = list()
    false_alarms = list()
    taus = list(np.linspace(.1, 1.5, 100))
    taus.append(1)
    one_idx = 0
    taus.sort()
    for tau in tqdm(taus):
    # for tau in [.5]:
        for i in range((M-8)):
            for j in range((N-8)):
                block = img[i:i+8, j:j+8] # This is a 8x8 block
                block = block.flatten(order='C')[:,None]

                likelihood_1 = get_likelihood(mu_1, sigma_1, sigma_1_inv, log_sigma_1, log_pi_1, block)
                likelihood_0 = get_likelihood(mu_0, sigma_0, sigma_0_inv, log_sigma_0, log_pi_0, block)

                if likelihood_1 / likelihood_0 > tau:
                    prediction[i, j] = 1
                else:
                    prediction[i, j] = 0
        
        # Count true positives
        tot_positives_in_ground_truth = 0
        tot_negs_in_ground_truth = 0
        true_positives = 0
        for i in range((M-8)):
            for j in range((N-8)):
                if prediction[i, j] > .9 / 256 and expected[i, j] > .9 / 256:
                    true_positives += 1

                if expected[i, j] >.9 / 256:
                    tot_positives_in_ground_truth += 1
                elif prediction[i, j] <.1 / 256:
                    tot_negs_in_ground_truth += 1
        
        # Count false positives
        false_positives = 0
        for i in range((M-8)):
            for j in range((N-8)):
                if prediction[i, j] < .1 / 256 and expected[i, j] < .1 / 256:
                    false_positives += 1

        # print("True Positives: %d" % true_positives)
        # print("False Positives: %d" % false_positives)
        if tau == 1:
            one_idx = len(detections)
        detections.append(true_positives / tot_positives_in_ground_truth)
        false_alarms.append(false_positives / tot_negs_in_ground_truth)

    
    # plt.imshow(prediction, cmap='Greys',  interpolation='nearest')
    # plt.show()

    # print(detections)
    # print(false_alarms)
    plt.plot(false_alarms, detections)
    plt.plot(false_alarms[one_idx], detections[one_idx], 'ro')
    plt.show()


def partd(predict_img, ground_truth_img):
    train_cat = np.matrix(np.loadtxt('data/train_cat.txt', delimiter=','))
    train_grass = np.matrix(np.loadtxt('data/train_grass.txt', delimiter=','))

    X = np.zeros((train_cat.shape[0], train_cat.shape[1] + train_grass.shape[1]))
    y = np.zeros((train_cat.shape[1] + train_grass.shape[1]))

    for i in range(train_cat.shape[1]):
        for j in range(train_cat.shape[0]):
            X[j, i] = train_cat[j, i]
            y[i] = 1

    for i in range(train_grass.shape[1]):
        for j in range(train_grass.shape[0]):
            X[j, train_cat.shape[1] + i] = train_grass[j, i]
            y[train_cat.shape[1] + i] = -1

    X = np.transpose(X)
    print(X.shape)
    print(y.shape)


    paren = np.matmul(np.matrix.transpose(X), X)
    weights = np.matmul(np.matmul(np.linalg.inv(paren), np.matrix.transpose(X)), y)

    print(weights.shape)

    img = plt.imread(predict_img) / 255
    expected = plt.imread(ground_truth_img) / 255

    M,N = img.shape

    detections = list()
    false_alarms = list()
    taus = list(np.linspace(-1, 1, 100))
    taus.append(1)
    one_idx = 0
    taus.sort()
    for tau in tqdm(taus):
        prediction = np.zeros((M-8,N-8))
        for i in range((M-8)):
            for j in range((N-8)):
                block = img[i:i+8, j:j+8] # This is a 8x8 block
                block = block.flatten(order='F')[:,None]

                predict = np.matmul(np.transpose(block), weights)[0]

                if predict > tau:
                    prediction[i, j] = 1
                else:
                    prediction[i, j] = 0

        # plt.imshow(prediction, cmap='Greys',  interpolation='nearest')
        # plt.show()

        # Count true positives
        tot_positives_in_ground_truth = 0
        tot_negs_in_ground_truth = 0
        true_positives = 0
        for i in range((M-8)):
            for j in range((N-8)):
                if prediction[i, j] > .9 / 256 and expected[i, j] > .9 / 256:
                    true_positives += 1

                if expected[i, j] >.9 / 256:
                    tot_positives_in_ground_truth += 1
                elif expected[i, j] <.1 / 256:
                    tot_negs_in_ground_truth += 1
        
        # Count false positives
        false_positives = 0
        for i in range((M-8)):
            for j in range((N-8)):
                if prediction[i, j] > .9 / 256 and expected[i, j] < .1 / 256:
                    false_positives += 1

        # print("True Positives: %d" % true_positives)
        # print("False Positives: %d" % false_positives)
        if tau == 1:
            one_idx = len(detections)
        detections.append(true_positives / tot_positives_in_ground_truth)
        false_alarms.append(false_positives / tot_negs_in_ground_truth)

    plt.plot(false_alarms, detections)
    plt.plot(false_alarms[one_idx], detections[one_idx], 'ro')
    plt.savefig('part3-d.png')
    plt.show()

    

def get_mu_sigma(data):
    mu = np.mean(data, axis=1)
    tot = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[1]):
        paren = data[:, i] - mu
        tot += np.matmul(paren, np.transpose(paren))

    sigma = tot / data.shape[1]

    return mu, sigma

def main():
    predict_img = 'data/cat_grass.jpg'
    ground_truth_img = 'data/truth.png'
    partb_c(predict_img, ground_truth_img)
    # partd(predict_img, ground_truth_img)

if __name__ == '__main__':
    main()