import os
import json
import argparse
import numpy as np
import tensorflow as tf
import midi_encoder as me

from train_generative import build_generative_model
from train_classifier import preprocess_sentence

GENERATED_DIR = './generated'

def override_neurons(model, layer_idx, override, temp=1.0):
    h_state, c_state = model.get_layer(index=layer_idx).states

    c_state = c_state.numpy()
    for neuron, value in override.items():
        c_state[:,int(neuron)] = int(value)

    model.get_layer(index=layer_idx).states = (h_state, tf.Variable(c_state))

def calc_sampling_penalty(midi_idxs, choices, probs, penalty=0.25):
    penalties = np.zeros_like(choices, dtype=float)

    for i in range(len(choices)):
        if choices[i] in midi_idxs:
            penalties[i] = (probs[i] * penalty)

    return penalties

def sample_next(midi_so_far, predictions, k):
    top_k = tf.math.top_k(predictions, k)

    top_k_choices = top_k[1].numpy().squeeze()
    top_k_values = top_k[0].numpy().squeeze()

    penalties = calc_sampling_penalty(midi_so_far, top_k_choices, top_k_values)
    p_choices = tf.math.softmax(top_k_values - penalties).numpy()

    argmax = np.argsort(p_choices)[-1]
    predicted_id = top_k_choices[argmax]

    return predicted_id

def process_init_text(model, init_text, char2idx):
    model.reset_states()

    for c in init_text.split(" "):
        # Run a forward pass
        try:
            input_eval = tf.expand_dims([char2idx[c]], 0)
            predictions = model(input_eval)
        except KeyError:
            if c != "":
                print("Can't process char", s)

    return predictions

def generate_midi(model, char2idx, idx2char, init_text="", seq_len=256, k=3, layer_idx=-2, override={}):
    # Add front and end pad to the initial text
    init_text = preprocess_sentence(init_text)

    # Empty midi to store our results
    midi_generated = []

    # Process initial text
    predictions = process_init_text(model, init_text, char2idx)

    # Complete the intial text with seq_len symbols
    for i in range(0, seq_len, 16):
        # override sentiment neurons
        # override_neurons(model, layer_idx, override)

        midi_chunck = []
        for j in range(16):
            # Remove the batch dimension
            predictions = tf.squeeze(predictions, 0).numpy()

            # Sample using a categorical distribution over the top k midi chars
            predicted_id = sample_next(midi_generated[-16:] + midi_chunck, predictions, k)

            # Append it to generated midi
            midi_chunck.append(predicted_id)

            #Run a new forward pass
            input_eval = tf.expand_dims([predicted_id], 0)
            predictions = model(input_eval)

        # Concatenate generated chunck to the final midi
        midi_generated += midi_chunck

    #Convert predicted index to char
    midi_generated = [idx2char[idx] for idx in midi_generated]

    return init_text + " " + " ".join(midi_generated)

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_generator.py')
    parser.add_argument('--model', type=str, required=True, help="Checkpoint dir.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--units', type=int, required=True, help="LSTM units.")
    parser.add_argument('--layers', type=int, required=True, help="LSTM layers.")
    parser.add_argument('--seqinit', type=str, default="", help="Sequence init.")
    parser.add_argument('--seqlen', type=int, default=256, help="Sequence lenght.")
    parser.add_argument('--override', type=str, default="", help="JSON file with neurons to override.")
    parser.add_argument('--cellix', type=int, default=-2, help="LSTM layer to use as encoder.")
    opt = parser.parse_args()

    # Load char2idx dict from json file
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Load override dict from json file
    override = {}
    try:
        with open(opt.override) as f:
            override = json.load(f)
    except:
        print("Not overriding any neurons.")

    # Create idx2char from char2idx dict
    idx2char = {idx:char for char,idx in char2idx.items()}

    # Calculate vocab_size from char2idx dict
    vocab_size = len(char2idx)

    # Rebuild model from checkpoint
    model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(opt.model))
    model.build(tf.TensorShape([1, None]))

    # Generate a midi as text
    midi_txt = generate_midi(model, char2idx, idx2char, opt.seqinit, opt.seqlen, override=override, layer_idx=opt.cellix)
    print(midi_txt)

    me.write(midi_txt, os.path.join(GENERATED_DIR, "generated.mid"))
