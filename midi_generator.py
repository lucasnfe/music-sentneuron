import json
import argparse
import tensorflow as tf
import midi_encoder as me

from train_generative import build_generative_model

def generate_midi(model, start_string, sequence_length, char2idx, idx2char, temperature):
    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string.split(" ")]

    # Add the batch dimension
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    midi_generated = []

    # Here batch size == 1
    model.reset_states()
    for i in range(sequence_length):
        predictions = model(input_eval)

        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        midi_generated.append(idx2char[predicted_id])

    return start_string + " " + " ".join(midi_generated)

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_generator.py')
    parser.add_argument('--model', type=str, required=True, help="Checkpoint dir.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--units', type=int, required=True, help="LSTM units.")
    parser.add_argument('--layers', type=int, required=True, help="LSTM layers.")
    parser.add_argument('--seqinit', type=str, default="\n", help="Sequence init.")
    parser.add_argument('--seqlen', type=int, default=256, help="Sequence lenght.")
    parser.add_argument('--temp', type=float, default=1.0, help="Sampling temperature.")
    opt = parser.parse_args()

    # Load char2idx dict from json file
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Create idx2char from char2idx dict
    idx2char = {idx:char for char,idx in char2idx.items()}

    # Calculate vocab_size from char2idx dict
    vocab_size = len(char2idx)

    # Rebuild model from checkpoint
    model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(opt.model))
    model.build(tf.TensorShape([1, None]))

    # Generate a midi as text
    midi_txt = generate_midi(model, opt.seqinit, opt.seqlen, char2idx, idx2char, opt.temp)
    print(midi_txt)

    me.write(midi_txt, "generated.mid")
