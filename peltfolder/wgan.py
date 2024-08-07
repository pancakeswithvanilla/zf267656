
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


latent_dim = 100
output_dim = 100  
input_dim = 100
learning_rate = 5e-5
batch_size = 64
critic_iterations = 5
weight_clip = 0.01
num_epochs = 1001
max_value = 269.5
min_value = -0.015
num_samples = 10
frag_signal_events_file = "fragsigev.txt"
frag_signal_nonevents_file = "fragsignonev.txt"

def read_signals(file_name):
    signals = []
    with open(file_name, 'r') as file:
        for line in file:
            signal = list(map(float, line.strip().split()))
            signals.append(signal)
    return np.array(signals)


def prepare_data() -> Tuple[np.ndarray, np.ndarray]:
    signal_events = read_signals(frag_signal_events_file)
    signal_nonevents = read_signals(frag_signal_nonevents_file)
    x = np.vstack((signal_events, signal_nonevents))
    x_train, x_test = train_test_split(x, test_size=0.1, random_state=42)
    print("x_train, x_test share common elements",check_common_elements(x_train, x_test))
    return x_train, x_test

def check_common_elements(x_train: np.ndarray, x_test: np.ndarray) -> bool:
    train_set = set(map(tuple, x_train))
    test_set = set(map(tuple, x_test))
    common_elements = train_set.intersection(test_set)
    return len(common_elements) > 0 # returns False, so they dont share common elements

def create_batches(data: np.ndarray):
    n_samples = data.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    data = data[indices]
    n_batches = n_samples // batch_size
    for i in range(n_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]
        yield batch
    if n_samples % batch_size != 0:
        yield data[n_batches * batch_size:]

def build_generator():
    model = tf.keras.Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(output_dim, activation='tanh')
    ])
    
    noise = Input(shape=(latent_dim,))
    generated_data = model(noise)
    return Model(noise, generated_data)
def normalize_data(data):
    return 2 * ((data - min_value) / (max_value - min_value)) - 1

def denormalize_data(normalized_data):
    return ((normalized_data + 1) / 2) * (max_value - min_value) + min_value

def build_discriminator():
    model = tf.keras.Sequential([
        Dense(512, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    data = Input(shape=(input_dim,))
    validity = model(data)
    return Model(data, validity)

def plot_losses(gen_losses, critic_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(gen_losses)), gen_losses, label='Generator Loss')
    plt.plot(range(len(critic_losses)), critic_losses, label='Critic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

generator = build_generator()
critic = build_discriminator()
opt_gen = tf.keras.optimizers.RMSprop(learning_rate)
opt_critic = tf.keras.optimizers.RMSprop(learning_rate)


fixed_noise = np.random.normal(0, 1, (32, latent_dim))

critic_losses = []
generator_losses = []


x_train, x_test = prepare_data()
x_train_normalized = normalize_data(x_train)
x_test_normalized = normalize_data(x_test)
print("Number of x_test samples",x_test.shape[0])
print("Number of x_train samples",x_train.shape[0])
save_path =f'losseswgan{num_epochs}{critic_iterations}_plot.png'
plots_directory = "/work/zf267656/peltfolder/plots"
file_path = os.path.join(plots_directory, save_path)

def save_signals_to_txt(file_name, signals):
    directory = "/work/zf267656/peltfolder/generated_signals"
    os.makedirs(directory, exist_ok=True)  
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'w') as file:
        for signal in signals:
            line = ' '.join(map(str, signal))
            file.write(line + '\n\n')

def save_accuracy_metrics(file_name, real_correct, fake_correct, real_acc, fake_acc, critic_iterations):
    with open(file_name, 'a') as file:
        file.write(f"{critic_iterations}\n")
        file.write(f"Number of real samples correctly predicted as real: {real_correct}\n")
        file.write(f"Number of fake samples correctly predicted as fake: {fake_correct}\n")
        file.write(f"Real_accuracy: {real_acc}\n")
        file.write(f"Fake_accuracy: {fake_acc}\n")

def evaluate_discriminator_performance(batch, critic_iterations):
    noise = np.random.normal(0, 1, (batch.shape[0], latent_dim))
    generated_samples = generator(noise, training=False)
    critic_real = critic(batch, training=False)
    critic_fake = critic(generated_samples, training=False)
    real_labels = tf.ones_like(critic_real, dtype=tf.int32)
    fake_labels = tf.zeros_like(critic_fake, dtype=tf.int32)
    print("Real labels", critic_real)
    print("Fake labels", critic_fake)
    real_predictions = tf.cast(critic_real < 0.5, tf.int32) #changed sign
    fake_predictions = tf.cast(critic_fake >= 0.5, tf.int32)
    real_accuracy = tf.reduce_mean(tf.cast(real_predictions == real_labels, tf.float32)).numpy()
    fake_accuracy = tf.reduce_mean(tf.cast(fake_predictions == fake_labels, tf.float32)).numpy()
    total_accuracy = (real_accuracy + fake_accuracy) / 2
    real_correct_predictions = tf.reduce_sum(tf.cast(real_predictions == real_labels, tf.int32)).numpy()
    fake_correct_predictions = tf.reduce_sum(tf.cast(fake_predictions == fake_labels, tf.int32)).numpy()
    file_name = "discriminator_accuracy.txt"
    save_accuracy_metrics(file_name, real_correct_predictions, fake_correct_predictions, real_accuracy, fake_accuracy,critic_iterations)
    return real_accuracy, fake_accuracy



def train(critic_iterations):
    num_epochs = 1000
    x_train, x_test = prepare_data()
    x_train_normalized = normalize_data(x_train)
    x_test_normalized = normalize_data(x_test)
    save_path =f'losseswgan{num_epochs}{critic_iterations}_plot.png'
    plots_directory = "/work/zf267656/peltfolder/plots"
    file_path = os.path.join(plots_directory, save_path)
    for epoch in range(num_epochs):
        print("Epoch Number: ", epoch)
        data_batches = create_batches(x_train_normalized)
        for _ in range(critic_iterations):
            try:
                data_batch = next(data_batches)
            except StopIteration:
                data_batches = create_batches(x_train_normalized, batch_size)
                data_batch = next(data_batches)
            total_critic_loss = 0
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            with tf.GradientTape() as crit_tape:
                fake = generator(noise, training=True)
                critic_real = critic(data_batch, training=True)
                critic_fake = critic(fake, training=True)
                loss_critic = -(tf.reduce_mean(critic_real) - tf.reduce_mean(critic_fake))
                total_critic_loss += loss_critic.numpy()
            critic_grads = crit_tape.gradient(loss_critic, critic.trainable_variables)
            opt_critic.apply_gradients(zip(critic_grads, critic.trainable_variables))

            for var in critic.trainable_variables:
                var.assign(tf.clip_by_value(var, -weight_clip, weight_clip))
        
        average_critic_loss = total_critic_loss / critic_iterations
        critic_losses.append(average_critic_loss)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        with tf.GradientTape() as gen_tape:
            gen_fake = generator(noise, training=True)
            critic_gen_fake = critic(gen_fake, training=True)
            loss_gen = -tf.reduce_mean(critic_gen_fake)
        generator_losses.append(loss_gen)
        gen_grads = gen_tape.gradient(loss_gen, generator.trainable_variables)
        opt_gen.apply_gradients(zip(gen_grads, generator.trainable_variables))
        if epoch == num_epochs - 1:
            noise = np.random.normal(0, 1, (num_samples, latent_dim))
            generated_samples = generator(noise, training=False)
            denorm_generated_samples = denormalize_data(generated_samples.numpy())
            save_signals_to_txt(f'generated_samples_epoch_{epoch + 1}{critic_iterations}.txt', denorm_generated_samples)

    plot_losses(generator_losses, critic_losses, save_path)
    real_accuracy, fake_accuracy = evaluate_discriminator_performance(x_test_normalized, critic_iterations)
    return real_accuracy, fake_accuracy 
    
    
#train()
