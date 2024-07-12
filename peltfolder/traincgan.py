import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU, Embedding, Concatenate, Masking
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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

# Split the datasets into training and testing sets
signal_events_train, signal_events_test = train_test_split(signal_events, test_size=0.2, random_state=42)
signal_nonevents_train, signal_nonevents_test = train_test_split(signal_nonevents, test_size=0.2, random_state=42)

# Padding masking
def create_masked_dataset(data):
    # Create a mask to ignore padding (0s at the end)
    mask = np.not_equal(data, 0).astype(np.float32)
    return tf.data.Dataset.from_tensor_slices((data, mask))

signal_events_dataset = create_masked_dataset(signal_events_train)
signal_nonevents_dataset = create_masked_dataset(signal_nonevents_train)

# CGAN Parameters
latent_dim = 100
input_dim = 11000

# Generator model
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

# Discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Dense(512, input_shape=(input_dim + input_dim,)))  # Correct input shape
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
    signal = Input(shape=(input_dim,))
    signal_label = Input(shape=(1,), dtype='int32')
    signal_embedding = Flatten()(Embedding(2, input_dim)(signal_label))
    signal_embedding_resized = Reshape((input_dim,))(signal_embedding)
    model_input = Concatenate()([signal, signal_embedding_resized])
    
    validity = model(model_input)
    
    return Model([signal, signal_label], validity)

# Compile the models
optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Build the generator
generator = build_generator()

# The generator takes noise and the target label as input and generates a fake signal
noise = Input(shape=(latent_dim,))
signal_label = Input(shape=(1,), dtype='int32')
generated_signal = generator([noise, signal_label])

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated signal and the label as input and determines validity
validity = discriminator([generated_signal, signal_label])

# The combined model (stacked generator and discriminator)
combined = Model([noise, signal_label], validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# Training the CGAN
def train(epochs, batch_size=128, save_interval=50):
    X_train = np.vstack([signal_events_train, signal_nonevents_train])
    y_train = np.array([1] * len(signal_events_train) + [0] * len(signal_nonevents_train))
    
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_signals, labels = X_train[idx], y_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_labels = np.random.randint(0, 2, batch_size)
        generated_signals = generator.predict([noise, gen_labels])
        
        d_loss_real = discriminator.train_on_batch([real_signals, labels], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([generated_signals, gen_labels], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.array([1] * batch_size)
        
        g_loss = combined.train_on_batch([noise, gen_labels], valid_y)
        
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")
        
        if epoch % save_interval == 0:
            save_model(epoch)
    # Evaluate performance on the testing set
    evaluate_performance(X_train, y_train)

def save_model(epoch):
    generator.save(f"generator_{epoch}.h5")
    discriminator.save(f"discriminator_{epoch}.h5")
    combined.save(f"combined_{epoch}.h5")

def evaluate_performance(X_test, y_test):
    y_pred = []
    for signal in X_test:
        label = np.argmax(discriminator.predict([signal.reshape(1, -1), np.array([0])]))
        y_pred.append(label)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

# Train the CGAN
train(epochs=1000, batch_size=32, save_interval=1000)
