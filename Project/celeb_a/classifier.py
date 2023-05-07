from cond_gan_celeba import *


CHECKPOINT_DIR = 'classifier_checkpoint'

def load_imgs(data_pool):
    img_fpaths = [data_pool[i][0] for i in range(len(data_pool))]

    imgs = list()
    for img_fpath in tqdm(img_fpaths, desc="Reading images from files..."):
        img = Image.open(img_fpath)
        img = img.resize((IM_SHAPE[0], IM_SHAPE[1]))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(IM_SHAPE)
        imgs.append(img)

    ys = np.array([[data_pool[i][1][0], data_pool[i][1][2]] for i in range(len(data_pool))])
    imgs = np.array(imgs)
    return imgs, ys

def define_classifier():
    inputs_img = layers.Input(shape=(32, 32, 3))

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='relu')(inputs_img)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)

    x = layers.Flatten()(x)

    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    out = layers.Dense(2, activation='sigmoid')(x)

    model= keras.Model(inputs_img, out, name="latent_discriminator_model")
    model.summary()

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'mean_absolute_error'])

    return model

def train_classifier():
    classifier = define_classifier()
    train_pool, test_pool = load_data_pool('../../data/Project/celeba_cropped_labels.csv', '../../data/Project/celebA_cropped/')
    train_xs, train_ys = load_imgs(train_pool)
    test_xs, test_ys = load_imgs(test_pool)

    print(train_xs.shape)
    print(train_ys.shape)

    classifier.fit(train_xs, train_ys, epochs=5, batch_size=128, validation_data=(test_xs, test_ys), shuffle=True)
    classifier.save('classifier.h5')

    predictions = classifier.predict(test_xs)
    predictions = np.round(predictions)

    male_correct = 0
    smiling_correct = 0
    both_correct = 0
    for row in range(test_xs.shape[0]):
        if predictions[row][0] == test_ys[row][0]:
            male_correct += 1
        if predictions[row][1] == test_ys[row][1]:
            smiling_correct += 1
        if predictions[row][1] == test_ys[row][1] and predictions[row][0] == test_ys[row][0]:
            both_correct += 1

    print("Male accuracy: %.4f" % (male_correct / test_xs.shape[0]))
    print("Smiling accuracy: %.4f" % (smiling_correct / test_xs.shape[0]))
    print("Both accuracy: %.4f" % (both_correct / test_xs.shape[0]))

    ''' Ouput:
        Male accuracy: 0.9687
        Smiling accuracy: 0.9093
        Both accuracy: 0.8810
    '''

def test_classifier():
    classifier = keras.models.load_model('classifier.h5')
    gan = GAN.load('trials/gan_labels_3_stopped_2')
    train_pool, test_pool = load_data_pool('../../data/Project/celeba_cropped_labels.csv', '../../data/Project/celebA_cropped/')
    
    # get labels from test pool
    latent = list()
    labels_gan = list()
    labels_classifier = list()
    for i in range(len(test_pool)):
        
        labels_gan.append(test_pool[i][1])
        labels_classifier.append(test_pool[i][1][[0,2]])
    
    latent = np.random.normal(0, 0.5, 96*len(labels_gan)).reshape((len(labels_gan), 96))
    labels_gan = np.array(labels_gan)
    labels_classifier = np.array(labels_classifier)

    print(latent.shape)
    print(labels_gan.shape)
    print(labels_classifier.shape)

    imgs = gan.generate_full_img(latent, labels_gan, verbose=1)
    predictions = classifier.predict(imgs)
    predictions = np.round(predictions)

    male_correct = 0
    smiling_correct = 0
    both_correct = 0
    for row in range(imgs.shape[0]):
        if predictions[row][0] == labels_classifier[row][0]:
            male_correct += 1
        if predictions[row][1] == labels_classifier[row][1]:
            smiling_correct += 1
        if predictions[row][1] == labels_classifier[row][1] and predictions[row][0] == labels_classifier[row][0]:
            both_correct += 1

    print("Male accuracy: %.4f" % (male_correct / labels_classifier.shape[0]))
    print("Smiling accuracy: %.4f" % (smiling_correct / labels_classifier.shape[0]))
    print("Both accuracy: %.4f" % (both_correct / labels_classifier.shape[0]))

    '''Output:
        Male accuracy: 0.9653
        Smiling accuracy: 0.8882
        Both accuracy: 0.8581
    '''


def main():

    test_classifier()
    

if __name__ == '__main__':
    main()