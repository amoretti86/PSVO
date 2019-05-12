import tensorflow as tf
import math


class Attention(tf.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, attention_dropout):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of heads.")

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="output_transform")

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.
        The tensor is transposed to insure the inner dimensions hold the correc
        values during the matrix multiplication.
        Args:
            x: A tensor with shape [batch_size, length, hidden_size]
        Returns:
            A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.
        Args:
            x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
        Returns:
            A tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y):
        """Apply attention mechanism to x and y.
        Args:
            x: a tensor with shape [batch_size, length_x, hidden_size]
            y: a tensor with shape [batch_size, length_y, hidden_size]
            bias: attention bias that will be added to the result of the dot product.
            cache: (Used during prediction) dictionary with tensors containing results
                of previous attentions. The dictionary must have the items:
                        {"k": tensor with shape [batch_size, i, key_channels],
                         "v": tensor with shape [batch_size, i, value_channels]}
                where i is the current decoded length.
        Returns:
            Attention layer output with shape [batch_size, length_x, hidden_size]
        """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(logits, name="attention_weights")
        weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
        attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x):
        return super(SelfAttention, self).call(x, x)


class FeedFowardNetwork(tf.layers.Layer):
    """Fully connected feedforward network."""

    def __init__(self, hidden_size, filter_size, relu_dropout):
        super(FeedFowardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout

        self.filter_dense_layer = tf.layers.Dense(
            filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
        self.output_dense_layer = tf.layers.Dense(
            hidden_size, use_bias=True, name="output_layer")

    def call(self, x):
        """Return outputs of the feedforward network.
        Args:
            x: tensor with shape [batch_size, length, hidden_size]
        Returns:
            Output of the feedforward network.
            tensor with shape [batch_size, length, hidden_size]
        """
        # Retrieve dynamically known shapes
        output = self.filter_dense_layer(x)
        output = tf.nn.dropout(output, 1.0 - self.relu_dropout)
        output = self.output_dense_layer(output)

        return output


class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                    initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, hidden_size, postprocess_dropout):
        self.layer = layer
        self.postprocess_dropout = postprocess_dropout

        # Create normalization layer
        self.layer_norm = LayerNormalization(hidden_size)

    def __call__(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
        return x + y


def get_position_encoding(
        length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.
    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.
    Args:
        length: Sequence length.
        hidden_size: Size of the
        min_timescale: Minimum scale that will be applied at each position
        max_timescale: Maximum scale that will be applied at each position
    Returns:
        Tensor with shape [length, hidden_size]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


class AttentionStack(tf.layers.Layer):
    """Transformer encoder stack.
    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
        1. Self-attention layer
        2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self, num_hidden_layers, hidden_size, num_heads, filter_size, dropout):
        super(AttentionStack, self).__init__()
        self.layers = []
        self.hidden_size = hidden_size
        for _ in range(num_hidden_layers):
            # Create sublayers for each layer.
            self_attention_layer = SelfAttention(hidden_size, num_heads, dropout)
            feed_forward_network = FeedFowardNetwork(hidden_size, filter_size, dropout)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, hidden_size, dropout),
                PrePostProcessingWrapper(feed_forward_network, hidden_size, dropout)])

        self.input_dense_layer = tf.layers.Dense(
            hidden_size, use_bias=True, name="input_layer")
        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(hidden_size)

    def call(self, encoder_inputs):
        """Return the output of the encoder layer stacks.
        Args:
            encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
            attention_bias: bias for the encoder self-attention layer.
                [batch_size, 1, 1, input_length]
            inputs_padding: P
        Returns:
            Output of encoder layer stack.
            float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        encoder_inputs = self.input_dense_layer(encoder_inputs)
        encoder_inputs += get_position_encoding(encoder_inputs.shape[1], self.hidden_size)
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.variable_scope("layer_%d" % n):
                with tf.variable_scope("self_attention"):
                    encoder_inputs = self_attention_layer(encoder_inputs)
                with tf.variable_scope("ffn"):
                    encoder_inputs = feed_forward_network(encoder_inputs)

        return self.output_normalization(encoder_inputs)
