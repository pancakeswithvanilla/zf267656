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

def read_signals(file_name):
    signals = []
    with open(file_name, 'r') as file:
        for line in file:
            signal = list(map(float, line.strip().split()))
            signals.append(signal)
    return np.array(signals)


def prepare_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:


    # Load data
    signal_events = read_signals(signal_events_file)
    signal_nonevents = read_signals(signal_nonevents_file)

    # Create lists to store formatted data
    # Create labels and ensure they are 1D
    labels_events = torch.ones(signal_events.size(0), dtype=torch.float32)  # Label 1 for events
    labels_nonevents = torch.zeros(signal_nonevents.size(0), dtype=torch.float32)  # Label 0 for non-events
    data_events_list = [(signal_events[i].tolist(), labels_events[i].item()) for i in range(signal_events.size(0))]
    data_nonevents_list = [(signal_nonevents[i].tolist(), labels_nonevents[i].item()) for i in range(signal_nonevents.size(0))]

    # Combine event and non-event data
    combined_data_list = data_events_list + data_nonevents_list

    # Convert the combined data to a tensor
    # Convert list of tuples to list of lists where each item is [[features], label]
    combined_data_tensor = torch.tensor([item[0] + [item[1]] for item in combined_data_list], dtype=torch.float32)

    # Split data into training and testing sets
    data_train, data_test = train_test_split(combined_data_tensor, test_size=0.2, random_state=42)
    print(data_train[0])

   

def build_discriminator():
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(input_dim,)))
    model.add(Dense(512))  # Correct input shape
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.4)) read about dropout layers
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    signal = Input(shape=(input_dim,))
    signal_label = Input(shape=(1,), dtype='int32')
    signal_embedding = Flatten()(Embedding(2, input_dim)(signal_label))
    signal_embedding_resized = Reshape((input_dim,))(signal_embedding)
    model_input = Concatenate()([signal, signal_embedding_resized])

    validity = model(model_input)

    return Model([signal, signal_label], validity)


def main():
    signals = read_signals(signal_events_file)
    data_3d_np = signals[:, np.newaxis,np.newaxis,  :]
    print(data_3d_np)



if __name__ == "__main__":
    main()