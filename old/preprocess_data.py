from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import re

## Load text data from a file
file_path = 'your_file.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

## Tokenization
sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s|\n', text) if sentence.strip()]
max_sequence_length = 6

# Initialize Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1

## Create inputs and targets
words = []
for sentence in sentences:
    for word in sentence.split():
        if (len(word)>0):
            words.append(word)

sequences = []
targets = []
for k in range(1, max_sequence_length+1):
    for i in range(len(words) - max_sequence_length):
        sequences.append(words[i:i+k])
        targets.append(words[i+k])

inputs = tokenizer.texts_to_sequences(sequences)
inputs = pad_sequences(inputs, maxlen=max_sequence_length, padding='pre')

targets = tokenizer.texts_to_sequences(targets)
targets = pad_sequences(targets, maxlen=1, padding='pre')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    inputs, targets, test_size=0.2, random_state=42
)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# Define the batch size
batch_size = 2048

# Shuffle and batch the dataset
train_dataset = train_dataset.shuffle(buffer_size=len(X_train), seed=42)
train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)