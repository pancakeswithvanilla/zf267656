import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU, Embedding, Concatenate, Masking
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.losses import binary_crossentropy
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
import matplotlib.pyplot as plt
import os

input_dim = 100
signal_events_file = "signalevents.txt"
signal_nonevents_file = "signalnonevents.txt"
frag_signal_events_file = "fragsigev.txt"
frag_signal_nonevents_file = "fragsignonev.txt"
batch_size = 64
dropout_rate = 0.2
latent_dim = 50 # see how model performs for different latent dim values
epochs = 1000
max_value = 269.25
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
    signal_events = read_signals(frag_signal_events_file)
    signal_nonevents = read_signals(frag_signal_nonevents_file)
    
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
    signal_input = Input(shape=(input_dim,))
    signal_label_input = Input(shape=(1,), dtype='int32')
    signal_embedding = Flatten()(Embedding(2, latent_dim)(signal_label_input))
    model_input = Concatenate()([signal_input, signal_embedding])
    masked_input = Masking(mask_value=0.0)(model_input)
    
    model = Sequential([
        Dense(512, input_shape=(input_dim + latent_dim,)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(dropout_rate),
        Dense(256),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    validity = model(model_input)
    
    return Model([signal_input, signal_label_input], validity) 

def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim * 2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(Dropout(dropout_rate))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(input_dim, activation='sigmoid')) #values between 0 and 1 
    model.add(Reshape((input_dim,)))
    
    noise = Input(shape=(latent_dim,))
    signal_label = Input(shape=(1,), dtype='int32')
    signal_embedding = Flatten()(Embedding(2, latent_dim)(signal_label))
    model_input = Concatenate()([noise, signal_embedding])
    
    signal = model(model_input)
    
    return Model([noise, signal_label], signal)

def save_signals_to_txt(file_name, signals):
    with open(file_name, 'w') as file:
        for signal in signals:
            line = ' '.join(map(str, signal))
            file.write(line + '\n\n')

def plot_losses(d_losses, g_losses, save_path):
    epochs = range(1, len(d_losses) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, d_losses, label='Discriminator Loss', color='blue')
    plt.plot(epochs, g_losses, label='Generator Loss', color='red')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Discriminator and Generator Losses')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(save_path)
    plt.close()


def train():
    d_losses = []
    g_losses = []
    save_path ='losses_plot.png'
    plots_directory = "plots/"
    file_path = os.path.join(plots_directory, save_path)
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
    x_train_normalized = normalize_data(x_train)
    x_test_normalized = normalize_data(x_test)
    # x_test = x_test.astype(np.float32)
    # y_test = y_test.astype(np.float32) # event vs nonevent
    # assert not np.any(np.isnan(x_test)), "x_test contains NaNs"
    # assert not np.any(np.isinf(x_test)), "x_test contains Infs"
    # assert not np.any(np.isnan(y_test)), "y_test contains NaNs"
    # assert not np.any(np.isinf(y_test)), "y_test contains Infs"
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
    validity = discriminator([generated_signal, label]) # fake or real sample
    
    cgan = Model([noise, label], validity)
    cgan.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    for epoch in range(epochs):
        # Train the discriminator
        idx = np.random.randint(0, x_train_normalized.shape[0], batch_size)
        real_signals, labels = x_train_normalized[idx], y_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_labels = np.random.randint(0, 2, (batch_size, 1)) # event or not event
        fake_signals = generator.predict([noise, fake_labels])
        
        d_loss_real = discriminator.train_on_batch([real_signals, labels], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([fake_signals, fake_labels], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_labels = np.random.randint(0, 2, (batch_size, 1))
        g_loss = cgan.train_on_batch([noise, fake_labels], np.ones((batch_size, 1)))
        d_losses.append(d_loss[0])
        g_losses.append(g_loss[0])
        if epoch == epochs - 1:
            generated_samples = generator.predict([np.random.normal(0, 1, (16, latent_dim)), np.random.randint(0, 2, (16, 1))])
            generated_samples_denorm = denormalize_data(generated_samples)
            
            # Save generated samples to a text file
            save_signals_to_txt(f'generated_samples_epoch_{epoch + 1}.txt', generated_samples_denorm)

    noise = np.random.normal(0, 1, (x_test.shape[0], latent_dim))
    fake_labels = np.random.randint(0, 2, (x_test.shape[0], 1))
    fake_signals = generator.predict([noise, fake_labels])
    print("Fake and real samples classification")
    evaluate_discriminator_performance(discriminator, x_test_normalized, y_test, fake_signals, fake_labels)
    print("Real samples predicted as Event/Nonevent")
    evaluate_samples(discriminator,x_test_normalized, y_test)
    print("Fake samples predicted as Event/Nonevent")
    evaluate_samples(discriminator, fake_signals, fake_labels)
    plot_losses(d_losses, g_losses, file_path)

def evaluate_samples(model, x_data, y_data):
    # Ensure y_data contains both data and labels
    assert x_data.shape[0] == y_data.shape[0], "Mismatch between x_data and y_data samples"
    
    # Extract signals and labels
    x_data_signals = x_data
    x_data_labels = y_data
    
    # Perform predictions
    predictions = model.predict([x_data_signals, x_data_labels])  # Predict if real or fake samples
    
    # Calculate loss
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    loss = loss_fn(x_data_labels, predictions).numpy()
    
    # Calculate accuracy
    predictions_binary = (predictions > 0.5).astype(np.float32)
    accuracy = np.mean(predictions_binary == x_data_labels)
    
    # Metrics for real events vs. real nonevents
    real_event_mask = (x_data_labels == 1)
    real_nonevent_mask = (x_data_labels == 0)
    
    real_event_predictions = predictions_binary[real_event_mask]
    real_nonevent_predictions = predictions_binary[real_nonevent_mask]
    
    # Calculate metrics for real events
    real_event_labels = x_data_labels[real_event_mask]
    real_event_pred_event = np.mean(real_event_predictions == real_event_labels)  # Correctly predicted events
    real_event_pred_nonevent = np.mean(real_event_predictions != real_event_labels)  # Incorrectly predicted events
    
    # Calculate metrics for real nonevents
    real_nonevent_labels = x_data_labels[real_nonevent_mask]
    real_nonevent_pred_nonevent = np.mean(real_nonevent_predictions == real_nonevent_labels)  # Correctly predicted nonevents
    real_nonevent_pred_event = np.mean(real_nonevent_predictions != real_nonevent_labels)  # Incorrectly predicted nonevents
    
    # Print out metrics
    print(f"Overall Loss: {loss:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Real\Fake Event Predicted as Event: {real_event_pred_event:.4f}")
    print(f"Real\Fake Event Predicted as Nonevent: {real_event_pred_nonevent:.4f}")
    print(f"Real\Fake Nonevent Predicted as Nonevent: {real_nonevent_pred_nonevent:.4f}")
    print(f"Real\Fake Nonevent Predicted as Event: {real_nonevent_pred_event:.4f}")



def evaluate_discriminator_performance(discriminator, x_real, y_real, x_fake, y_fake):
    """
    Evaluate the performance of the discriminator by predicting and comparing real vs. fake samples.

    Args:
        discriminator (tf.keras.Model): The trained discriminator model.
        x_real (np.ndarray): Real signal samples.
        y_real (np.ndarray): Labels for real signal samples (event or nonevent).
        x_fake (np.ndarray): Fake signal samples generated by the generator.
        y_fake (np.ndarray): Labels for fake signal samples (event or nonevent).
    """
    # Predict if samples are real or fake
    real_predictions = discriminator.predict([x_real, y_real])
    fake_predictions = discriminator.predict([x_fake, y_fake])

    # Convert predictions to binary (real or fake)
    real_predictions_binary = (real_predictions > 0.5).astype(np.float32)
    fake_predictions_binary = (fake_predictions > 0.5).astype(np.float32)

    # Define masks for real and fake samples
    real_mask = (y_real == 1)
    fake_mask = (y_fake == 0)

    # Metrics for real samples
    real_pred_real = real_predictions_binary[real_mask]  # Predictions for real samples
    real_pred_fake = real_predictions_binary[~real_mask]  # Predictions for fake samples

    # Metrics for fake samples
    fake_pred_real = fake_predictions_binary[real_mask]  # Predictions for real samples classified as fake
    fake_pred_fake = fake_predictions_binary[~real_mask]  # Predictions for fake samples classified as real

    # Calculate how many samples were predicted as fake or real correctly or incorrectly
    real_predicted_fake_as_real = np.sum(real_predictions_binary[real_mask] == 0)
    real_predicted_fake_as_fake = np.sum(real_predictions_binary[real_mask] == 1)
    fake_predicted_real_as_fake = np.sum(fake_predictions_binary[fake_mask] == 1)
    fake_predicted_real_as_real = np.sum(fake_predictions_binary[fake_mask] == 0)

    print(f"Real samples predicted as fake: {real_predicted_fake_as_real}")
    print(f"Real samples predicted as real: {real_predicted_fake_as_fake}")
    print(f"Fake samples predicted as real: {fake_predicted_real_as_real}")
    print(f"Fake samples predicted as fake: {fake_predicted_real_as_fake}")

def normalize_data(data):
    return data / max_value

def denormalize_data(normalized_data):
    return normalized_data * max_value

def main():
    train()

if __name__ == "__main__":
    main()