import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU, Embedding, Concatenate, Masking
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.losses import binary_crossentropy
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn

LENGTH_INPUT = 11000
# Load data
signal_events_file = "signalevents.txt"
signal_nonevents_file = "signalnonevents.txt"

def read_signals(file_name):
    signals = []
    with open(file_name, 'r') as file:
        for line in file:
            signal = list(map(float, line.strip().split()))
            signals.append(signal)
    return np.array(signals)

signal_events = read_signals(signal_events_file)
signal_nonevents = read_signals(signal_nonevents_file)

def generate_batches_of_signals(batch_size, data_tensor):
    batch_indices = np.random.choice(len(data_tensor), size=batch_size, replace=False)
    batch = [data_tensor[idx] for idx in batch_indices]
    signals_batch, labels_batch = zip(*batch)
    x_real = torch.stack(signals_batch)
    y_real = torch.stack(labels_batch)
    return x_real, y_real #test this

def generate_latent_points(latent_dim, n):
    x_input = randn(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim, n)
    X = generator.predict(x_input, verbose=0)
    y = zeros((n, 1))
    return X, y

def prepare_data():
    signal_nonevents = read_signals(signal_nonevents_file)
    signal_events_train, signal_events_test = train_test_split(signal_events, test_size=0.2, random_state=42)
    signal_nonevents_train, signal_nonevents_test = train_test_split(signal_nonevents, test_size=0.2, random_state=42)
    signal_events_train = [(torch.tensor(signal), torch.tensor(1)) for signal in signal_events_train]
    signal_events_test = [(torch.tensor(signal), torch.tensor(1)) for signal in signal_events_test]
    signal_nonevents_train = [(torch.tensor(signal), torch.tensor(0)) for signal in signal_nonevents_train]
    signal_nonevents_test = [(torch.tensor(signal), torch.tensor(0)) for signal in signal_nonevents_test]
    train_data = signal_events_train + signal_nonevents_train
    test_data = signal_events_test + signal_nonevents_test
    train_data_tensor = [(torch.unsqueeze(signal, 0), label) for signal, label in train_data]
    test_data_tensor = [(torch.unsqueeze(signal, 0), label) for signal, label in test_data]
    return train_data_tensor, test_data_tensor

def define_discriminator(n_inputs=LENGTH_INPUT):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(n_inputs,)))
    model.add(Dense(250))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    model.add(generator)
    model.add(Reshape((LENGTH_INPUT, 1)))  # Reshape back to 2D for discriminator
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=64, n_eval=200):
    half_batch = int(n_batch / 2)
    train_data_tensor, test_data_tensor = prepare_data()
    for i in range(n_epochs):
        x_real, y_real = generate_batches_of_signals(half_batch, train_data_tensor)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        gan_model.train_on_batch(x_gan, y_gan)
        # evaluate the model every n_eval epochs
        if (i+1) % n_eval == 0:
            plt.title('Number of epochs = %i'%(i+1))
            pred_data = generate_fake_samples(generator,latent_dim,latent_dim)[0]
            real_data  = generate_real_samples(latent_dim)[0]
            plt.plot(pred_data[0],'.',label='Random Fake Sample',color='firebrick')
            plt.plot(real_data[0],'.',label = 'Random Real Sample',color='navy')
            plt.legend(fontsize=10)
            plt.show()