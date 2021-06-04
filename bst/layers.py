import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

from deepctr.layers.utils import reduce_sum, reduce_max, reduce_mean, softmax


class PositionalEncoding(Layer):
    '''Sinusoidal Positional_Encoding.

    Args:

      - inputs: A 2d Tensor with shape of (N, T).
      - num_units: Output dimensionality
      - zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      - scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      - scope: Optional scope for `variable_scope`.
      - reuse: Boolean, whether to reuse the weights of a previous layer by the same name.

    Returns:

      - A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    def __init__(self,
                 max_len,
                 num_hidden,
                 dropout_rate=None,
                 pos_embedding_trainable=True,
                 zero_pad=False,
                 scale=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate
        self.pos_embedding_trainable = pos_embedding_trainable
        self.zero_pad = zero_pad
        self.scale = scale
        if self.dropout_rate:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.P = np.zeros((1, max_len, num_hidden))
        X = np.arange(max_len, dtype=np.float32).reshape(-1, 1) / np.power(
            10000,
            np.arange(0, num_hidden, 2, dtype=np.float32) / num_hidden)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        if self.pos_embedding_trainable:
            self.lookup_table = K.variable(self.P, dtype=tf.float32)
            if self.zero_pad:
                self.lookup_table = tf.concat(
                    (tf.zeros(shape=[1, 1, num_hidden]),
                     self.lookup_table[:, 1:, :]), 1)
        else:
            self.lookup_table = tf.convert_to_tensor(self.P, dtype=tf.float32)
        if self.scale:
            self.lookup_table = self.lookup_table * num_hidden**0.5

    def call(self, inputs, training=None):
        pos = self.lookup_table[:, :inputs.shape[1], :]
        outputs = pos + inputs
        if self.dropout_rate:
            outputs = self.dropout(outputs, training=training)
        return outputs


class Transformer(Layer):
    """  Simplified version of Transformer  proposed in 《Attention is all you need》

      Input shape
        - a list of two 3D tensor with shape ``(batch_size, timesteps, input_dim)`` if ``supports_masking=True`` .
        - a list of two 4 tensors, first two tensors with shape ``(batch_size, timesteps, input_dim)``,last two tensors with shape ``(batch_size, 1)`` if ``supports_masking=False`` .


      Output shape
        - 3D tensor with shape: ``(batch_size, 1, input_dim)``  if ``output_type='mean'`` or ``output_type='sum'`` , else  ``(batch_size, timesteps, input_dim)`` .


      Arguments
            - **att_embedding_size**: int.The embedding size in multi-head self-attention network.
            - **head_num**: int.The head number in multi-head  self-attention network.
            - **dropout_rate**: float between 0 and 1. Fraction of the units to drop.
            - **use_positional_encoding**: bool. Whether or not use positional_encoding
            - **use_res**: bool. Whether or not use standard residual connections before output.
            - **use_feed_forward**: bool. Whether or not use pointwise feed foward network.
            - **use_layer_norm**: bool. Whether or not use Layer Normalization.
            - **blinding**: bool. Whether or not use blinding.
            - **seed**: A Python integer to use as random seed.
            - **supports_masking**:bool. Whether or not support masking.
            - **attention_type**: str, Type of attention, the value must be one of { ``'scaled_dot_product'`` , ``'additive'`` }.
            - **output_type**: ``'mean'`` , ``'sum'`` or `None`. Whether or not use average/sum pooling for output.

      References
            - [Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
    """

    def __init__(self,
                 att_embedding_size=1,
                 head_num=8,
                 dropout_rate=0.0,
                 use_positional_encoding=True,
                 use_res=True,
                 use_feed_forward=True,
                 use_layer_norm=False,
                 blinding=True,
                 seed=1024,
                 supports_masking=False,
                 attention_type="scaled_dot_product",
                 output_type="mean",
                 **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.num_units = att_embedding_size * head_num
        self.use_res = use_res
        self.use_feed_forward = use_feed_forward
        self.seed = seed
        self.use_positional_encoding = use_positional_encoding
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.blinding = blinding
        self.attention_type = attention_type
        self.output_type = output_type
        super(Transformer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        embedding_size = int(input_shape[0][-1])
        if self.num_units != embedding_size:
            raise ValueError(
                "att_embedding_size * head_num must equal the last dimension size of inputs,got %d * %d != %d"
                % (self.att_embedding_size, self.head_num, embedding_size))
        self.seq_len_max = int(input_shape[0][-2])
        self.W_Query = self.add_weight(
            name='query',
            shape=[embedding_size, self.att_embedding_size * self.head_num],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(
            name='key',
            shape=[embedding_size, self.att_embedding_size * self.head_num],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed +
                                                              1))
        self.W_Value = self.add_weight(
            name='value',
            shape=[embedding_size, self.att_embedding_size * self.head_num],
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed +
                                                              2))
        if self.attention_type == "additive":
            self.b = self.add_weight(
                'b',
                shape=[self.att_embedding_size],
                dtype=tf.float32,
                initializer=tf.keras.initializers.glorot_uniform(
                    seed=self.seed))
            self.v = self.add_weight(
                'v',
                shape=[self.att_embedding_size],
                dtype=tf.float32,
                initializer=tf.keras.initializers.glorot_uniform(
                    seed=self.seed))
        # if self.use_res:
        #     self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num], dtype=tf.float32,
        #                                  initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        if self.use_feed_forward:
            self.fw1 = self.add_weight(
                'fw1',
                shape=[self.num_units, 4 * self.num_units],
                dtype=tf.float32,
                initializer=tf.keras.initializers.glorot_uniform(
                    seed=self.seed))
            self.fw2 = self.add_weight(
                'fw2',
                shape=[4 * self.num_units, self.num_units],
                dtype=tf.float32,
                initializer=tf.keras.initializers.glorot_uniform(
                    seed=self.seed))

        if self.use_positional_encoding:
            self.pe = PositionalEncoding(self.seq_len_max, self.num_units)

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate,
                                               seed=self.seed)
        self.ln = tf.keras.layers.LayerNormalization()
        # Be sure to call this somewhere!
        super(Transformer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):

        if self.supports_masking:
            queries, keys = inputs
            query_masks, key_masks = mask
            query_masks = tf.cast(query_masks, tf.float32)
            key_masks = tf.cast(key_masks, tf.float32)
        else:
            queries, keys, query_masks, key_masks = inputs

            query_masks = tf.sequence_mask(query_masks,
                                           self.seq_len_max,
                                           dtype=tf.float32)
            key_masks = tf.sequence_mask(key_masks,
                                         self.seq_len_max,
                                         dtype=tf.float32)
            query_masks = tf.squeeze(query_masks, axis=1)
            key_masks = tf.squeeze(key_masks, axis=1)

        if self.use_positional_encoding:
            queries = self.pe(queries)
            keys = self.pe(keys)

        querys = tf.tensordot(queries, self.W_Query,
                              axes=(-1, 0))  # None T_q D*head_num
        keys = tf.tensordot(keys, self.W_key, axes=(-1, 0))
        values = tf.tensordot(keys, self.W_Value, axes=(-1, 0))

        # head_num*None T_q D
        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0)

        if self.attention_type == "scaled_dot_product":
            # head_num*None T_q T_k
            outputs = tf.matmul(querys, keys, transpose_b=True)

            outputs = outputs / (keys.get_shape().as_list()[-1]**0.5)
        elif self.attention_type == "additive":
            querys_reshaped = tf.expand_dims(querys, axis=-2)
            keys_reshaped = tf.expand_dims(keys, axis=-3)
            outputs = tf.tanh(
                tf.nn.bias_add(querys_reshaped + keys_reshaped, self.b))
            outputs = tf.squeeze(tf.tensordot(outputs,
                                              tf.expand_dims(self.v, axis=-1),
                                              axes=[-1, 0]),
                                 axis=-1)
        else:
            NotImplementedError

        key_masks = tf.tile(key_masks, [self.head_num, 1])

        # (h*N, T_q, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1),
                            [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2**32 + 1)

        # (h*N, T_q, T_k)

        outputs = tf.where(
            tf.equal(key_masks, 1),
            outputs,
            paddings,
        )
        if self.blinding:
            try:
                outputs = tf.matrix_set_diag(
                    outputs,
                    tf.ones_like(outputs)[:, :, 0] * (-2**32 + 1))
            except:
                outputs = tf.compat.v1.matrix_set_diag(
                    outputs,
                    tf.ones_like(outputs)[:, :, 0] * (-2**32 + 1))

        outputs -= reduce_max(outputs, axis=-1, keep_dims=True)
        outputs = softmax(outputs)
        query_masks = tf.tile(query_masks, [self.head_num, 1])  # (h*N, T_q)
        # (h*N, T_q, T_k)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1),
                              [1, 1, tf.shape(keys)[1]])

        outputs *= query_masks

        outputs = self.dropout(outputs, training=training)
        # Weighted sum
        # ( h*N, T_q, C/h)
        result = tf.matmul(outputs, values)
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=2)

        if self.use_res:
            # tf.tensordot(queries, self.W_Res, axes=(-1, 0))
            result += queries
        if self.use_layer_norm:
            result = self.ln(result)

        if self.use_feed_forward:
            fw1 = tf.nn.relu(tf.tensordot(result, self.fw1, axes=[-1, 0]))
            fw1 = self.dropout(fw1, training=training)
            fw2 = tf.tensordot(fw1, self.fw2, axes=[-1, 0])
            if self.use_res:
                result += fw2
            if self.use_layer_norm:
                result = self.ln(result)

        if self.output_type == "mean":
            return reduce_mean(result, axis=1, keep_dims=True)
        elif self.output_type == "sum":
            return reduce_sum(result, axis=1, keep_dims=True)
        else:
            return result

    def compute_output_shape(self, input_shape):

        return (None, 1, self.att_embedding_size * self.head_num)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self, ):
        config = {
            'att_embedding_size': self.att_embedding_size,
            'head_num': self.head_num,
            'dropout_rate': self.dropout_rate,
            'use_res': self.use_res,
            'use_positional_encoding': self.use_positional_encoding,
            'use_feed_forward': self.use_feed_forward,
            'use_layer_norm': self.use_layer_norm,
            'seed': self.seed,
            'supports_masking': self.supports_masking,
            'blinding': self.blinding,
            'attention_type': self.attention_type,
            'output_type': self.output_type
        }
        base_config = super(Transformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
