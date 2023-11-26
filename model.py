import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, ReLU, LayerNormalization, MultiHeadAttention, Dropout, Embedding

## Feed Forward Layer ##
class FeedForward(Layer):
    def __init__(self, dim_inter, d_model, **kwargs):
        ##**kwargs is a variable number of arguments
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = Dense(dim_inter)
        self.fully_connected2 = Dense(d_model)
        self.activation = ReLU()

    def call(self, inputs):
        # Expected input shape = (batch_size, sequence_length, d_model)
        # For each token in the sequence we flatten its embbeding to create d_model units
        x = self.fully_connected1(inputs)
        x = self.activation(x)
        x = self.fully_connected2(x)
        # Expected output shape = (batch_size, sequence_length, d_model)
        return x
    
## Add & Norm Layer ##
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()

    def call(self, x, y):
        # Expected input shape = (batch_size, sequence_length, d_model)
        add = x + y
        # Expected output shape = (batch_size, sequence_length, d_model)
        return self.layer_norm(add)

## Encoder Layer ##
class EncoderLayer(Layer):
    def __init__(self, h, dim_key, dim_value, d_model, d_ff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(num_heads = h, key_dim = dim_key, value_dim = dim_value)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def call(self, x, padding_mask, training):
        # Multi-head attention layer, expected input x of shape (batch_size, sequence_length, d_model)
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)

## Positional Encoding embedding Layer ##
def positional_encoding(seq_length, d_model):
    d_model = d_model // 2
    positions = np.arange(seq_length)[:, np.newaxis]
    d_models = np.arange(d_model)[np.newaxis, :] / d_model

    angle_rates = 1 / (10000**d_models)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEncodingEmbedding(Layer):
    def __init__(self, max_seq_len, vocab_size, d_model, **kwargs):
        super(PositionalEncodingEmbedding, self).__init__(**kwargs)
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        # Expected input shape = (batch_size, sequence_length)
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        # Expected output shape = (batch_size, sequence_length, d_model)
        return x

## Transformer Encoder ##    
class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, number_heads, dim_key, dim_value, d_model, dim_inter, number_layers, rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.pos_encoding = PositionalEncodingEmbedding(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(number_heads, dim_key, dim_value, d_model, dim_inter, rate) for _ in range(number_layers)]

    def call(self, inputs, padding_mask, training):
        # Generate the positional encoding, Expected input shape = (batch_size, sequence_length)
        pos_encoding_output = self.pos_encoding(inputs)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each encoder layer
        for layer in self.encoder_layer:
            x = layer(x, padding_mask, training)

        return x
    
## Decoder Layer ##
class DecoderLayer(Layer):
    def __init__(self, num_heads, dim_key, dim_value, d_model, dim_inter, rate, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(num_heads = num_heads, key_dim = dim_key, value_dim = dim_value)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(dim_inter, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def call(self, decoder_output, x, padding_mask, training):
        # Multi-head attention layer, expected input x of shape (batch_size, sequence_length, d_model)
        multihead_output = self.multihead_attention(query=x, value=decoder_output, key=decoder_output, attention_mask=padding_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)
    
## Transformer decoder ##
class Decoder(Layer):
    def __init__(self, vocab_size, sequence_length, number_heads, dim_key, dim_value, d_model, dim_inter, number_layers, rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.pos_encoding = PositionalEncodingEmbedding(sequence_length, vocab_size, d_model)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.masked_multihead_attention = MultiHeadAttention(num_heads = number_heads, key_dim = dim_key, value_dim = dim_value)
        self.add_norm1 = AddNormalization()
        self.decoder_layers = [DecoderLayer(number_heads, dim_key, dim_value, d_model, dim_inter, rate) for _ in range(number_layers)]

    def call(self, encoder_output, inputs, padding_mask, training):
        # Generate the positional encoding, Expected input shape = (batch_size, sequence_length)
        pos_encoding_output = self.pos_encoding(inputs)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        x = self.dropout1(pos_encoding_output, training=training)

        # Multi-head attention layer, expected input x of shape (batch_size, sequence_length, d_model)
        masked_multihead_output = self.masked_multihead_attention(x, x, x, padding_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        masked_multihead_output = self.dropout2(masked_multihead_output, training=training)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, masked_multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Pass on the positional encoded values to each encoder layer
        for layer in self.decoder_layers:
            addnorm_output = layer(encoder_output, addnorm_output, padding_mask, training)

        # Expected output shape = (batch_size, sequence_length, d_model)
        return addnorm_output
    
## Final transformer model ##
class Transformer(Layer):
    def __init__(self, vocab_size, sequence_length, number_heads, dim_key, dim_value, d_model, dim_inter, number_layers, rate, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.encoder = Encoder(vocab_size, sequence_length, number_heads, dim_key, dim_value, d_model, dim_inter, number_layers, rate)
        self.decoder = Decoder(vocab_size, 1, number_heads, dim_key, dim_value, d_model, dim_inter, number_layers, rate)
        # Linear layer
        self.linear_layer = Dense(vocab_size)

    def call(self, src, tgt, padding_mask, training):
        encoder_output = self.encoder(src, padding_mask, training)
        decoder_output = self.decoder(encoder_output, tgt, padding_mask, training)
        # Apply the linear layer to the decoder output
        logits = self.linear_layer(decoder_output)

        # Softmax activation
        softmax_probs = tf.nn.softmax(logits)

        return softmax_probs