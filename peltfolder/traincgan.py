import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU, Embedding, Concatenate, Masking
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.losses import binary_cross_entropy, conditional_cross_entropy

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
# masking seems to work fine
# def print_masked_sample(dataset, num_samples=5):
#     for i, (data, mask) in enumerate(dataset.take(num_samples)):
#         print(f"Sample {i+1}")
#         print("Data:", data.numpy())
#         print("Mask:", mask.numpy())
#         print()

# # Print samples from signal_events_dataset
# print("Signal Events Dataset Samples:")
# print_masked_sample(signal_events_dataset)

# # Print samples from signal_nonevents_dataset
# print("Signal Nonevents Dataset Samples:")
# print_masked_sample(signal_nonevents_dataset)



# def masked_accuracy(y_true, y_pred, mask):
#     accuracy = tf.cast(tf.equal(y_true, tf.round(y_pred)), tf.float32)
#     masked_accuracy = tf.reduce_sum(accuracy * mask) / tf.reduce_sum(mask)
#     return masked_accuracy


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
generator_optimizer = Adam(0.0002, 0.5)
discriminator_optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_discriminator()
# discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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
# combined.compile(loss='binary_crossentropy', optimizer=optimizer)
def masked_binary_crossentropy(y_true, y_pred, mask):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
    masked_loss = loss * mask
    
    return masked_loss

@tf.function
def train_step(real_signals, labels, masks):
    batch_size = real_signals.shape[0]

    # Generate noise and labels for fake signals
    noise = tf.random.normal([batch_size, latent_dim])
    gen_labels = tf.random.uniform([batch_size, 1], minval=0, maxval=2, dtype=tf.int32)

    with tf.GradientTape() as tape:
        generated_signals = generator([noise, gen_labels], training=True)

        # Compute discriminator loss on real and fake signals
        real_validity = tf.ones((batch_size, 1))
        fake_validity = tf.zeros((batch_size, 1))

        real_predictions = discriminator([real_signals, labels], training=True)
        fake_predictions = discriminator([generated_signals, gen_labels], training=True)
        sum_loss_real = 0
        sum_loss_fake = 0
        for index in range(len(masks)):
            d_loss_real = masked_binary_crossentropy(real_validity[index], real_predictions[index], masks[index])
            d_loss_fake = masked_binary_crossentropy(fake_validity[index], fake_predictions[index], masks[index])
            sum_loss_fake += d_loss_fake
            sum_loss_real += d_loss_real
        
        d_loss = 0.5 * (sum_loss_real + sum_loss_fake)

    gradients_of_discriminator = tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Train generator
    noise = tf.random.normal([batch_size, latent_dim])
    misleading_labels = tf.ones((batch_size, 1))  # Generator tries to fool the discriminator

    with tf.GradientTape() as tape:
        generated_signals = generator([noise, gen_labels], training=True)
        fake_predictions = discriminator([generated_signals, gen_labels], training=True)
        g_loss = masked_binary_crossentropy(misleading_labels, fake_predictions, masks)

    gradients_of_generator = tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return d_loss, g_loss

# Training the CGAN
def train(epochs, batch_size=128, save_interval=50):
    X_test = np.vstack([signal_events_test, signal_nonevents_test])
    y_test = np.array([1] * len(signal_events_test) + [0] * len(signal_nonevents_test))
    
    mask_test = np.vstack([np.not_equal(signal_events_test, 0).astype(np.float32),
                           np.not_equal(signal_nonevents_test, 0).astype(np.float32)])
    for epoch in range(epochs):
        # Get a batch of real signals and their masks
        signal_events_batch = list(signal_events_dataset.batch(batch_size).take(1))[0]
        signal_nonevents_batch = list(signal_nonevents_dataset.batch(batch_size).take(1))[0]

        real_signals_events, masks_events = signal_events_batch
        real_signals_nonevents, masks_nonevents = signal_nonevents_batch

        real_signals = tf.concat([real_signals_events, real_signals_nonevents], axis=0)
        labels = tf.concat([tf.ones(batch_size), tf.zeros(batch_size)], axis=0)
        masks = tf.concat([masks_events, masks_nonevents], axis=0)

        print("Masks first element:",masks[0]) #each mask element has 11000 values and I send batches of 64 masks
        print(len(masks))


        d_loss, g_loss = train_step(real_signals, labels, masks)

        print(f"{epoch} [D loss: {d_loss.numpy()}] [G loss: {g_loss.numpy()}]")

        if epoch % save_interval == 0:
            save_model(epoch)
    # Evaluate performance on the testing set
    evaluate_performance(X_test, y_test, mask_test)

def save_model(epoch):
    generator.save(f"generator_{epoch}.h5")
    discriminator.save(f"discriminator_{epoch}.h5")
    combined.save(f"combined_{epoch}.h5")

# def evaluate_performance(X_test, y_test, mask_test):
#     y_pred = []
#     for signal in X_test:
#         label = np.argmax(discriminator.predict([signal.reshape(1, -1), np.array([0])]))
#         y_pred.append(label)
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
#     print(f"Accuracy: {accuracy}")
#     print(f"Classification Report:\n{report}")

def evaluate_performance(X_test, y_test, mask_test):
    predictions = discriminator.predict([X_test, y_test])
    loss = masked_binary_crossentropy(np.ones(len(X_test)), predictions, mask_test)
    accuracy = np.mean((predictions > 0.5).astype(int) == y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy * 100}%")

# Train the CGAN
train(epochs=1000, batch_size=32, save_interval=1000)


def discriminator_loss(real_output, fake_output, real_labels, fake_labels):
    adversarial_loss = binary_cross_entropy(real_output, real_labels) + binary_cross_entropy(fake_output, fake_labels)
    conditional_loss = conditional_cross_entropy(real_output, real_labels) + conditional_cross_entropy(fake_output, fake_labels)
    return adversarial_loss + conditional_loss

# Generator loss
def generator_loss(fake_output, real_labels):
    adversarial_loss = binary_cross_entropy(fake_output, real_labels)
    conditional_loss = conditional_cross_entropy(fake_output, real_labels)
    return adversarial_loss + conditional_loss # check the conditional gan paper again