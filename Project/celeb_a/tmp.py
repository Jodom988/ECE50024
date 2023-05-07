from tensorflow.keras import layers
from tensorflow import keras

NOISE_DIM = 96
N_LABELS = 4

def build_generator():

    leakyrelu_alpha = 0.01

    inputs_img = layers.Input(shape=(32, 32, 3))
    inputs_labels = layers.Input(shape=(N_LABELS,))

    x = layers.Conv2D(128, (2,2), padding='same')(inputs_img)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(256, (2,2), padding='same')(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(512, (2,2), padding='same')(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(512, (2,2), padding='same')(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, inputs_labels])

    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)

    out = layers.Dense(1, activation='sigmoid')(x)
    
    model= keras.Model((inputs_img, inputs_labels), out, name="latent_discriminator_model")

    keras.utils.plot_model(model, show_shapes=True, dpi=64, to_file='figures/celeba_discriminator_full_model.png')


build_generator()