import os
import json
import argparse
import numpy      as np
import tensorflow as tf

from midi_encoder import MIDIEncoder

def build_model(vocab_size, embed_dim, lstm_units, batch_size):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(vocab_size, embed_dim, batch_input_shape=[batch_size, None]))
    model.add(tf.keras.layers.LSTM(lstm_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform", dropout=0.05, recurrent_dropout=0.05))
    model.add(tf.keras.layers.LSTM(lstm_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform", dropout=0.05, recurrent_dropout=0.05))
    model.add(tf.keras.layers.Dense(vocab_size))

    return model

def build_dataset(text, char2idx, seq_length, batch_size, buffer_size=10000):
    text_as_int = np.array([char2idx[c] for c in text.split(" ")])
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    dataset = sequences.map(__split_input_target)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    return dataset

def train_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def train_model(model, train_dataset, test_dataset, epochs, learning_rate):
    # Create Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile model with Adam optimizer and crossentropy Loss funciton
    model.compile(optimizer=optimizer, loss=train_loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    return model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=[checkpoint_callback])

def __split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='train_generative.py')
    parser.add_argument('--train', type=str, required=True, help="Train dataset.")
    parser.add_argument('--test' , type=str, required=False, help="Test dataset.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--units', type=int, required=True, help="LSTM units.")
    parser.add_argument('--batch', type=int, required=True, help="Batch size.")
    parser.add_argument('--epochs', type=int, required=True, help="Epochs.")
    parser.add_argument('--seqlen', type=int, required=True, help="Sequence lenght.")
    parser.add_argument('--lrate', type=float, required=True, help="Learning rate.")
    opt = parser.parse_args()

    # Encode midi files as text with vocab
    encoded_midis_train = MIDIEncoder(opt.train)
    encoded_midis_test = MIDIEncoder(opt.test)

    # Merge train and test vocabulary
    vocab = list(encoded_midis_train.vocab | encoded_midis_test.vocab)
    vocab.sort()

    # Calculate vocab size
    vocab_size = len(vocab)

    # Create dictionaries to support symbol to index conversion and vice-versa
    char2idx = { char:i for i,char in enumerate(vocab) }
    idx2char = { i:char for i,char in enumerate(vocab) }

    # Save char2idx encoding as a json file for generate midi later
    with open("char2idx.json", "w") as f:
        json.dump(char2idx, f)

    # Build dataset from encoded midis
    train_dataset = build_dataset(encoded_midis_train.text, char2idx, opt.seqlen, opt.batch)
    test_dataset = build_dataset(encoded_midis_test.text, char2idx, opt.seqlen, opt.batch)

    # Build model
    model = build_model(vocab_size, opt.embed, opt.units, opt.batch)

    # Train model
    history = train_model(model, train_dataset, test_dataset, opt.epochs, opt.lrate)