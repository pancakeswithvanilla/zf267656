import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU, Embedding, Concatenate, Masking
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.losses import binary_crossentropy
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn

input_dim = 11000
signal_events_file = "signalevents.txt"
signal_nonevents_file = "signalnonevents.txt"
batch_size = 64
latent_dim = 50 # see how model performs for different latent dim values
epochs = 1000
save_interval = 100
def read_signals(file_name):
    signals = []
    with open(file_name, 'r') as file:
        for line in file:
            signal = list(map(float, line.strip().split()))
            signals.append(signal)
    return np.array(signals)


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load data
    signal_events = read_signals(signal_events_file)
    signal_nonevents = read_signals(signal_nonevents_file)
    
    # Get the number of samples for events and nonevents
    num_events = signal_events.shape[0]
    num_nonevents = signal_nonevents.shape[0]
    
    # Create labels: 1 for events, 0 for nonevents
    labels_events = np.ones((num_events, 1), dtype=np.float32)
    labels_nonevents = np.zeros((num_nonevents, 1), dtype=np.float32)
    
    # Concatenate data and labels
    x = np.vstack((signal_events, signal_nonevents))
    y = np.vstack((labels_events, labels_nonevents))
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test


    

def build_discriminator():
    # Define the input layers
    signal_input = Input(shape=(input_dim,))
    signal_label_input = Input(shape=(1,), dtype='int32')

    # Define embedding for the label
    signal_embedding = Flatten()(Embedding(2, latent_dim)(signal_label_input))
    
    # Concatenate the signal and label embedding
    model_input = Concatenate()([signal_input, signal_embedding])
    
    # Masking layer: apply mask to the concatenated input if needed
    # This is useful if you have padded sequences, but in this case, it's added for completeness.
    masked_input = Masking(mask_value=0.0)(model_input)
    
    # Create the model
    model = Sequential([
        Dense(512, input_shape=(input_dim + latent_dim,)),  # Ensure correct input shape
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Pass the masked input through the model
    validity = model(model_input)
    
    return Model([signal_input, signal_label_input], validity)

def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim * 2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(input_dim, activation='tanh'))
    model.add(Reshape((input_dim,)))
    
    noise = Input(shape=(latent_dim,))
    signal_label = Input(shape=(1,), dtype='int32')
    signal_embedding = Flatten()(Embedding(2, latent_dim)(signal_label))
    model_input = Concatenate()([noise, signal_embedding])
    
    signal = model(model_input)
    
    return Model([noise, signal_label], signal)


def main():
    # Sample data
    # discriminator = build_discriminator()
    # discriminator.summary()
    # input_dim = 11000
    # test_signal = np.random.rand(1, input_dim)  # Normal data
    # test_signal_with_mask = np.copy(test_signal)
    # test_signal_with_mask[0, 0:] = 0.0  # Mask half of the signal, ok if I mask everything I always get around 50% because no input, so it guesses randomly the label which is good

    # Labels (just as a placeholder, real vs. fake label doesn't matter here)
    # test_labels = np.array([[1]])

    # # Check model predictions
    # print("Prediction with full signal:", discriminator.predict([test_signal, test_labels]))
    # print("Prediction with masked signal:", discriminator.predict([test_signal_with_mask, test_labels]))

    x_train, x_test, y_train, y_test = prepare_data()
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    assert not np.any(np.isnan(x_test)), "x_test contains NaNs"
    assert not np.any(np.isinf(x_test)), "x_test contains Infs"
    assert not np.any(np.isnan(y_test)), "y_test contains NaNs"
    assert not np.any(np.isinf(y_test)), "y_test contains Infs"
    generator = build_generator()
    discriminator = build_discriminator()
    
    optimizer = Adam(0.0002, 0.5)
    
    # Compile the discriminator
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # Create the combined model
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    generated_signal = generator([noise, label])
    discriminator.trainable = False
    validity = discriminator([generated_signal, label])
    
    cgan = Model([noise, label], validity)
    cgan.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    for epoch in range(epochs):
        # Train the discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_signals, labels = x_train[idx], y_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_labels = np.random.randint(0, 2, (batch_size, 1))
        fake_signals = generator.predict([noise, fake_labels])
        
        d_loss_real = discriminator.train_on_batch([real_signals, labels], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([fake_signals, fake_labels], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_labels = np.random.randint(0, 2, (batch_size, 1))
        g_loss = cgan.train_on_batch([noise, fake_labels], np.ones((batch_size, 1)))
        
        if (epoch + 1) % save_interval == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], D Loss: {d_loss[0]}, D Accuracy: {100*d_loss[1]}, G Loss: {g_loss[0]}, G Accuracy: {100*g_loss[1]}')
            
            # Save models
            generator.save(f'generator_epoch_{epoch + 1}.h5')
            discriminator.save(f'discriminator_epoch_{epoch + 1}.h5')
            
            # Generate and save samples
            generated_samples = generator.predict([np.random.normal(0, 1, (16, latent_dim)), np.random.randint(0, 2, (16, 1))])
            # Save or visualize generated_samples
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
        # Evaluate discriminator performance on test data
    print(f"Any None in x_test: {np.any(np.isnan(x_test))}")
    print(f"Any None in y_test: {np.any(np.isnan(y_test))}")
    # Evaluate the discriminator on test data
    # dummy_x_test = np.random.rand(4, input_dim).astype(np.float32)
    # dummy_y_test = np.random.randint(0, 2, (4, 1)).astype(np.float32)

    loss, accuracy = custom_evaluate(discriminator,x_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

def custom_evaluate(model, x_data, y_data):
    # Ensure y_data contains both data and labels
    assert x_data.shape[0] == y_data.shape[0], "Mismatch between x_data and y_data samples"
    
    # Extract labels
    x_data_signals = x_data  # These are the signals
    x_data_labels = y_data    # These are the labels
    
    # Perform predictions
    predictions = model.predict([x_data_signals, x_data_labels])
    
    # Calculate loss
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    loss = loss_fn(x_data_labels, predictions).numpy()
    
    # Calculate accuracy
    predictions_binary = (predictions > 0.5).astype(np.float32)
    accuracy = np.mean(predictions_binary == x_data_labels)
    
    return loss, accuracy

if __name__ == "__main__":
    main()