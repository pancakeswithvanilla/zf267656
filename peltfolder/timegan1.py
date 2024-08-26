import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
def build_embedder(input_shape, hidden_dim):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.LSTM(hidden_dim, return_sequences=True),
        layers.LSTM(hidden_dim, return_sequences=False)
    ])
    return model

def build_recovery(output_shape, hidden_dim):
    model = models.Sequential([
        layers.InputLayer(input_shape=(hidden_dim,)),
        layers.RepeatVector(output_shape[0]),
        layers.LSTM(hidden_dim, return_sequences=True),
        layers.LSTM(output_shape[1], return_sequences=True)
    ])
    return model
def build_generator(z_dim, hidden_dim):
    model = models.Sequential([
        layers.InputLayer(input_shape=(z_dim,)),
        layers.RepeatVector(1),
        layers.LSTM(hidden_dim, return_sequences=True),
        layers.LSTM(hidden_dim, return_sequences=True)
    ])
    return model

def build_supervisor(hidden_dim):
    model = models.Sequential([
        layers.InputLayer(input_shape=(hidden_dim,)),
        layers.RepeatVector(1),
        layers.LSTM(hidden_dim, return_sequences=True)
    ])
    return model

def build_discriminator(input_shape, hidden_dim):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.LSTM(hidden_dim, return_sequences=True),
        layers.LSTM(hidden_dim, return_sequences=False),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def get_gan_losses():
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def d_loss_fn(real, fake):
        real_loss = bce(tf.ones_like(real), real)
        fake_loss = bce(tf.zeros_like(fake), fake)
        return real_loss + fake_loss

    def g_loss_fn(fake):
        return bce(tf.ones_like(fake), fake)

    def embedding_loss_fn(real, pred):
        return tf.reduce_mean(tf.losses.mean_squared_error(real, pred))

    return d_loss_fn, g_loss_fn, embedding_loss_fn

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()
embedding_optimizer = tf.keras.optimizers.Adam()

def read_signals(file_name):
    signals = []
    with open(file_name, 'r') as file:
        for line in file:
            signal = list(map(float, line.strip().split()))
            signals.append(signal)
    return np.array(signals)


def prepare_data() -> Tuple[np.ndarray, np.ndarray]:
    signal_events = read_signals(frag_signal_events_file)
    # signal_nonevents = read_signals(frag_signal_nonevents_file)
    # x = np.vstack((signal_events, signal_nonevents))
    x_train, x_test = train_test_split(signal_events, test_size=0.1, random_state=42)
    return x_train, x_test
def normalize_data(data):
    return 2 * ((data - min_value) / (max_value - min_value)) - 1

def denormalize_data(normalized_data):
    return ((normalized_data + 1) / 2) * (max_value - min_value) + min_value

def train_timegan(data, epochs=4000, batch_size=64, z_dim=100):
    x_train, x_test = prepare_data()
    x_train_normalized = normalize_data(x_train)
    x_test_normalized = normalize_data(x_test)
    save_path =f'losseswgan_only_events_{num_epochs}{critic_iterations}_plot.png'
    plots_directory = "/work/zf267656/peltfolder/plots"
    file_path = os.path.join(plots_directory, save_path)
    # Define model dimensions
    input_shape = (data.shape[1], data.shape[2])
    hidden_dim = data.shape[2]
    
    # Build models
    embedder = build_embedder(input_shape, hidden_dim)
    recovery = build_recovery(input_shape, hidden_dim)
    generator = build_generator(z_dim, hidden_dim)
    supervisor = build_supervisor(hidden_dim)
    discriminator = build_discriminator(input_shape, hidden_dim)
    
    d_loss_fn, g_loss_fn, embedding_loss_fn = get_gan_losses()
    
    for epoch in range(epochs):
        for _ in range(batch_size):
            # Train embedder and recovery networks
            with tf.GradientTape() as tape:
                h = embedder(data)
                x_tilde = recovery(h)
                e_loss = embedding_loss_fn(data, x_tilde)
            grads = tape.gradient(e_loss, embedder.trainable_variables + recovery.trainable_variables)
            embedding_optimizer.apply_gradients(zip(grads, embedder.trainable_variables + recovery.trainable_variables))
            
            # Train generator, supervisor, and discriminator networks
            z = tf.random.normal((batch_size, z_dim))
            h_hat = generator(z)
            h_hat_supervised = supervisor(h_hat)
            
            with tf.GradientTape() as tape:
                x_hat = recovery(h_hat_supervised)
                fake_logits = discriminator(x_hat)
                real_logits = discriminator(data)
                g_loss = g_loss_fn(fake_logits)
                d_loss = d_loss_fn(real_logits, fake_logits)
            g_grads = tape.gradient(g_loss, generator.trainable_variables + supervisor.trainable_variables)
            d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
            generator_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables + supervisor.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
            
        