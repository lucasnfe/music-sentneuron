import os
import csv
import json
import argparse
import numpy as np
import tensorflow as tf
import midi_encoder as me

from train_generative import build_generative_model

def encode_sentence(model, text, char2idx, layer_idx):
    # Reset LSTMs hidden and cell states
    model.reset_states()

    for s in text.split(" "):
        # Add the batch dimension
        try:
            input_eval = tf.expand_dims([char2idx[s]], 0)
            predictions = model(input_eval)
        except KeyError:
            print("Could not process char:", s)

    h_state, c_state = model.get_layer(index=layer_idx).states

    # remove the batch dimension
    c_state = tf.squeeze(c_state, 0)

    return c_state

def build_classifier_model(input_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_dim=input_size, activation='sigmoid',
                    kernel_regularizer=tf.keras.regularizers.l1(l=0.01),
                      bias_regularizer=tf.keras.regularizers.l1(l=0.01)))
    return model

def build_dataset(datapath, generative_model, char2idx, layer_idx):
    xs, ys = [], []

    csv_file = open(datapath, "r")
    data = csv.DictReader(csv_file)

    for row in data:
        label = int(row["label"])
        filepath = row["filepath"]

        data_dir = os.path.dirname(datapath)
        phrase_path = os.path.join(data_dir, filepath) + ".mid"
        encoded_path = os.path.join(data_dir, filepath) + ".npy"

        # Load midi file as text
        if os.path.isfile(encoded_path):
            encoding = tf.convert_to_tensor(np.load(encoded_path))
        else:
            text, vocab = me.load(phrase_path, transpose_range=1, stretching_range=1)

            # Encode midi text using generative lstm
            encoding = encode_sentence(generative_model, text, char2idx, layer_idx=layer_idx)

            # Save encoding in file to make it faster to load next time
            np.save(encoded_path, encoding)

        xs.append(encoding)
        ys.append(label)

    return xs, ys

def train_generative_model(model, train_dataset, test_dataset, epochs, learning_rate=0.001):
    # Create Adam optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # Compile model with Adam optimizer and crossentropy Loss funciton
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy())

    return model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=[checkpoint_callback])

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='train_classifier.py')
    parser.add_argument('--train', type=str, required=True, help="Train dataset.")
    parser.add_argument('--test' , type=str, required=True, help="Test dataset.")
    parser.add_argument('--model', type=str, required=True, help="Checkpoint dir.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--units', type=int, required=True, help="LSTM units.")
    parser.add_argument('--layers', type=int, required=True, help="LSTM layers.")
    parser.add_argument('--cellix', type=int, required=True, help="LSTM layer to use as encoder.")
    opt = parser.parse_args()

    # Load char2idx dict from json file
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Calculate vocab_size from char2idx dict
    vocab_size = len(char2idx)

    # Rebuild generative model from checkpoint
    generative_model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, 1)
    generative_model.load_weights(tf.train.latest_checkpoint(opt.model))
    generative_model.build(tf.TensorShape([1, None]))

    # Build dataset from encoded labelled midis
    train_dataset = build_dataset(opt.train, generative_model, char2idx, opt.cellix)
    test_dataset = build_dataset(opt.test, generative_model, char2idx)

    # Build classifier model
    classifier_model = build_classifier_model(opt.units)
