import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def define_classifier():
    inputs_img = layers.Input(shape=(28, 28, 1))

    x = layers.GaussianNoise(0.02)(inputs_img)
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)

    x = layers.Flatten()(x)

    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    out = layers.Dense(10, activation='softmax')(x)

    model= keras.Model(inputs_img, out, name="latent_discriminator_model")
    model.summary()

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy', 'mean_absolute_error'])

    return model

def train_classifier():
    classifier = define_classifier()
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


    classifier.fit(X_train, y_train, epochs=2, batch_size=128, validation_data=(X_test, y_test), shuffle=True)
    classifier.save('classifier.h5')

    classifier.evaluate(X_test, y_test)


    ''' Ouput:
        469/469 [==============================] - 4s 8ms/step - loss: 0.0505 - accuracy: 0.9850 - mean_absolute_error: 4.3737 - val_loss: 0.0521 - val_accuracy: 0.9841 - val_mean_absolute_error: 4.3630
    '''

def test_classifier():
    classifier = keras.models.load_model('classifier.h5')
    generator = keras.models.load_model('trial_2/generator.h5')
    (_, _), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    np.random.seed(0)
    latent_input = np.random.normal(0, 0.5, 100*X_test.shape[0]).reshape((X_test.shape[0], 100))
    labels_one_hot = keras.utils.to_categorical(y_test, 10)

    imgs = generator.predict([latent_input, labels_one_hot])

    classifier.evaluate(imgs, y_test)

    predictions = classifier.predict(imgs)
    correct = 0
    for row in range(predictions.shape[0]):
        prediction = np.argmax(predictions[row])
        if prediction == y_test[row]:
            correct += 1

    print("Percent correct: %.4f" % (correct / predictions.shape[0]))

    '''Output:
        Percent correct: 0.3113

    '''

def test_classifer_by_label():
    classifier = keras.models.load_model('classifier.h5')
    generator = keras.models.load_model('trial_2/generator.h5')
    (_, _), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_test_bins = {}
    y_test_bins = {}
    real_accs = []
    gen_accs = []

    for i in range(10):
        X_test_bins[i] = []
        y_test_bins[i] = []

    # sort testing data set into bins
    for row in range(X_test.shape[0]):
        X_test_bins[y_test[row]].append(X_test[row])
        y_test_bins[y_test[row]].append(y_test[row])

    for i in range(10):
        X_test_bins[i] = np.array(X_test_bins[i])
        y_test_bins[i] = np.array(y_test_bins[i])

        loss, acc, mae = classifier.evaluate(X_test_bins[i], y_test_bins[i])
        acc = acc * 100
        real_accs.append(acc)

    print(real_accs)

    n_samples = 2048
    np.random.seed(0)
    latent_input = np.random.normal(0, 0.5, n_samples*100).reshape((n_samples, 100))
    for i in range(10):
        labels_one_hot = np.zeros((n_samples, 10))
        labels_one_hot[:, i] = 1
        Xs = generator.predict([latent_input, labels_one_hot])
        ys = np.ones((n_samples,)) * i

        loss, acc, mae = classifier.evaluate(Xs, ys)
        acc = acc * 100
        gen_accs.append(acc)

    print(gen_accs)

    classes = list(range(10))
    class_accs = {
        'Real': real_accs,
        'Generated': gen_accs
    }

    x = np.arange(len(classes))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(6, 4))

    for attribute, measurement in class_accs.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt='%d', rotation=90)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Class Label')
    ax.set_xticks(x + width*.5, classes)
    # ax.legend(loc='upper right')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=2)
    
    ax.set_ylim(0, 110)
    
    plt.savefig('class_accs.png')
    plt.show()


def main():
    # train_classifier()
    # test_classifier()
    test_classifer_by_label()
    

if __name__ == '__main__':
    main()