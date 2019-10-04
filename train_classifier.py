import json
import argparse
import tensorflow as tf

from train_generative import build_generative_model

def encode_sentence(model, text, char2idx):
    # Here batch size == 1
    model.reset_states()
    print(model.get_layer(index=3).states)

    for s in text.split(" "):
        input_eval = tf.expand_dims([char2idx[s]], 0)
        predictions = model(input_eval)

    print(model.get_layer(index=3).states)

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='train_classifier.py')
    # parser.add_argument('--train', type=str, required=True, help="Train dataset.")
    # parser.add_argument('--test' , type=str, required=True, help="Test dataset.")
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

    model.summary()

    # Remove last layer because we want to represent labelled sentences using the lstm cell state
    # model.pop()

    encode_sentence(model, "n_74 t_144 n_38", char2idx)
