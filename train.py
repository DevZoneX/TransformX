import tensorflow as tf
import numpy as np
from model import Transformer
from preprocess_data import train_dataset, X_val, y_val

## Define and train the model ##
number_heads, dim_key, dim_value, d_model, dim_inter, number_layers, rate, vocab_size, max_sequence_length, batch_size = (
    3, 32, 32, 32, 64, 1, 0.2, 50000, 64, 2048
)
transformer = Transformer(vocab_size, max_sequence_length, number_heads, dim_key, dim_value, d_model, dim_inter, number_layers, rate)

# Define The loss function
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Define The optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Set up training loop
zeros_matrix = np.zeros((batch_size, 1))
zeros_matrix_val = np.zeros((X_val.shape[0], 1))

epoches = 50

for epoch in range(epoches):
    for batch, (inputs, targets) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            loss = criterion(targets, transformer(inputs, zeros_matrix, padding_mask=None, training=True))

            optimizer.apply_gradients(zip(tape.gradient(loss, transformer.trainable_variables), transformer.trainable_variables))

            if (batch%10 == 0):
                print(f'batch: {batch+1} , Loss {loss}')

    # Validation
    val_output = transformer(X_val, zeros_matrix_val, padding_mask=None, training=False)
    val_loss = criterion(y_val, val_output)

    print(f"Epoch: {epoch + 1}, Training Loss: {loss}, Validation Loss: {val_loss}")

print('training completed')
weights = transformer.get_weights()
np.save('transformer_weights.npy', weights)
print('Weights saved.')

## Make predictions ##
def predict(text):
    words = text.split()
    zero_matrix = np.zeros((1, 1))

    for i in range(10):
        sequences = tokenizer.texts_to_sequences(words)
        input = np.array(sequences).reshape(-1)

        # Pad the sequences at the beginning to ensure a fixed length
        padded_sentence = pad_sequences([input], maxlen=max_sequence_length, padding='pre')
        output = transformer(padded_sentence, zero_matrix, padding_mask=None, training=False)

        max_index = np.argmax(output, axis=-1)
        if max_index > 0:
            words.append(tokenizer.sequences_to_texts(max_index)[0])
        else:
            print(" ".join(words))
            return
    print(" ".join(words))