import tensorflow as tf
import numpy as np
import os

# Step 1: Load Shakespeare Text Data
path_to_file = tf.keras.utils.get_file("shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print(f"Length of text: {len(text)} characters")

# Step 2: Preprocess the Text
vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# Step 3: Create Input-Target Sequences
seq_length = 100
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)

# Step 4: Prepare Data for Training
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Step 5: Build the LSTM Model (Fixed for Keras 3+)
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None,), batch_size=batch_size),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True,
                             stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

# Step 6: Compile and Train the Model
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Step 7: Save Checkpoints (Fixed filepath for Keras 3+)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

# Step 8: Train the model
EPOCHS = 10
model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Step 9: Rebuild Model for Inference (batch_size=1)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
model.build(tf.TensorShape([1, None]))

# Step 10: Generate Text using the trained model
def generate_text(model, start_string, temperature=1.0, num_generate=1000):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    generated = []

    model.reset_states()
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        generated.append(idx2char[predicted_id])

    return start_string + ''.join(generated)

# Example usage
print(generate_text(model, start_string="To be, or not to be", temperature=0.8))
