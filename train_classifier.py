import os
import csv
import json
import argparse
import tensorflow as tf
import midi_encoder as me

from train_generative import build_generative_model

def encode_sentence(model, text, char2idx, layer_idx):
    # Reset LSTMs hidden and cell states
    model.reset_states()

    for s in text.split(" "):
        # Add the batch dimension
        input_eval = tf.expand_dims([char2idx[s]], 0)
        predictions = model(input_eval)

    return model.get_layer(index=layer_idx).states[1]

def build_dataset(datapath):
    csv_file = open(datapath, "r")
    data = csv.DictReader(csv_file)

    for row in data:
        label = row["label"]
        filepath = row["filepath"]

        data_dir = os.path.dirname(datapath)
        phrase_path = os.path.join(data_dir, filepath) + ".mid"
        encoding, vocab = me.load(phrase_path, transpose_range=1, stretching_range=1)
        print(encoding)

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
    opt = parser.parse_args()

    # Load char2idx dict from json file
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Calculate vocab_size from char2idx dict
    vocab_size = len(char2idx)

    # Rebuild model from checkpoint
    model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, 1)
    model.load_weights(tf.train.latest_checkpoint(opt.model))
    model.build(tf.TensorShape([1, None]))

    build_dataset(opt.train)
    build_dataset(opt.test)

    encoding = encode_sentence(model, "n_74 t_144 n_38", char2idx, 3)
