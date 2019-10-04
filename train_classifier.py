
def build_reprensetation_model(vocab_size, embed_dim, lstm_units, lstm_layers, batch_size, dropout=0):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(vocab_size, embed_dim, batch_input_shape=[batch_size, None]))

    for i in range(max(1, lstm_layers)):
        model.add(tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True, stateful=True, recurrent_initializer="glorot_uniform", dropout=dropout, recurrent_dropout=dropout))

    return model

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='train_classifier.py')
    parser.add_argument('--train', type=str, required=True, help="Train dataset.")
    parser.add_argument('--test' , type=str, required=True, help="Test dataset.")
    parser.add_argument('--model', type=str, required=True, help="Checkpoint dir.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--units', type=int, required=True, help="LSTM units.")
    parser.add_argument('--layers', type=int, required=True, help="LSTM layers.")
    opt = parser.parse_args()
