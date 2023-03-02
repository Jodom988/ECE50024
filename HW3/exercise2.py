import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm    

def get_likelihood(mu, sigma, sigma_inv, log_sigma, log_pi, x):
    x_minus_mu = x - mu
    x_minus_mu_t = np.transpose(x_minus_mu)

    term1 = -0.5 * np.matmul(np.matmul(x_minus_mu_t, sigma_inv), x_minus_mu)[0, 0]
    term2 = log_pi
    term3 = log_sigma

    # print(term1.shape)
    # print(term2.shape)
    # print(term3.shape)

    return term1 + term2 + term3

def partc_e(predict_img, ground_truth_img):
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
    M,N = img.shape
    prediction = np.zeros((M-8,N-8))
    for i in tqdm(range((M-8))):
        for j in range((N-8)):
            block = img[i:i+8, j:j+8] # This is a 8x8 block
            block = block.flatten(order='C')[:,None]

            likelihood_1 = get_likelihood(mu_1, sigma_1, sigma_1_inv, log_sigma_1, log_pi_1, block)
            likelihood_0 = get_likelihood(mu_0, sigma_0, sigma_0_inv, log_sigma_0, log_pi_0, block)

            if likelihood_1 > likelihood_0:
                prediction[i, j] = 1
            else:
                prediction[i, j] = 0
    
    plt.imshow(prediction, cmap='Greys',  interpolation='nearest')
    plt.show()
    
    if ground_truth_img is None:
        return 
    
    expected = plt.imread(ground_truth_img) / 255
    abs_error = 0
    for i in tqdm(range((M-8))):
        for j in range((N-8)):
            abs_error += abs(prediction[i, j] - expected[i, j])

    print("MAE: %.3f" % (abs_error / ((M-8)*(N-8))))

    

def get_mu_sigma(data):
    mu = np.mean(data, axis=1)
    tot = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[1]):
        paren = data[:, i] - mu
        tot += np.matmul(paren, np.transpose(paren))

    sigma = tot / data.shape[1]

    return mu, sigma

def partb():
    train_cat = np.matrix(np.loadtxt('data/train_cat.txt', delimiter=','))
    train_grass = np.matrix(np.loadtxt('data/train_grass.txt', delimiter=','))

    Y = plt.imread('data/cat_grass.jpg') / 255

    mu_1, sigma_1 = get_mu_sigma(train_cat)
    mu_0, sigma_0 = get_mu_sigma(train_grass)

    pi_1 = train_cat.shape[1] / (train_cat.shape[1] + train_grass.shape[1])
    pi_0 = train_grass.shape[1] / (train_cat.shape[1] + train_grass.shape[1])

    print("mu_1:\n%s\n" % str(mu_1[:2]))
    print("mu_0:\n%s\n" % str(mu_0[:2]))

    print("sigma_1:\n%s\n" % str(sigma_1[:2, :2]))
    print("sigma_0:\n%s\n" % str(sigma_0[:2, :2]))

    print("pi_1:\n%s\n" % str(pi_1))
    print("pi_0:\n%s\n" % str(pi_0))


def main():
    # partb()
    partc_e('data/cat_grass.jpg', 'data/truth.png')
    #partc_e('data/my_image.jpg', None)



if __name__ == '__main__':
    main()