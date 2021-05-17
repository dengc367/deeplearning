import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal


class MaskZero(Layer):
    """Set values to zeroes when the row is masked
    """

    def call(self, inputs, mask=None):
        if mask is None:
            return inputs
        mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
        return mask * inputs

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        return super().get_config()


class ListMeanPooling(Layer):
    def __init__(self, **kwargs):
        super(ListMeanPooling, self).__init__(**kwargs)
        self.epsilon = 1e-12

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, mask=None):
        if not isinstance(inputs, list):
            return inputs

        inputs = tf.stack(inputs, axis=0)

        if mask is None:
            return tf.reduce_mean(inputs, axis=0, keepdims=False)

        mask = tf.stack(mask, axis=0)
        mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)

        inputs_sum = tf.reduce_sum(inputs * mask, axis=0, keepdims=False)
        mask_sum = tf.reduce_sum(mask, axis=0, keepdims=False)
        mean = tf.divide(inputs_sum, tf.math.maximum(mask_sum, tf.constant(self.epsilon, dtype=tf.float32)))
        return mean

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def compute_mask(self, inputs, mask=None):
        if mask is None or not isinstance(mask, list):
            return mask
        mask = tf.stack(mask, axis=0)
        mask = tf.reduce_sum(tf.cast(mask, tf.float32), axis=0, keepdims=False)
        mask = tf.cast(mask, tf.bool)
        return mask

    def get_config(self):
        config = {
            'epsilon': self.epsilon
        }
        base_config = super(ListMeanPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SplitString(Layer):
    """
    String Splitter Layer
    """

    def __init__(self,
                 att_feats,
                 sep='_',
                 **kwargs):
        self.att_feats = att_feats
        self.attr_len = len(att_feats)
        self.sep = sep

        super(SplitString, self).__init__(**kwargs)

    def call(self, inputs):
        outputs = {}
        if self.attr_len <= 1:
            return outputs
        split_tensor = tf.strings.split(inputs, sep=self.sep).to_tensor(shape=inputs.shape + [self.attr_len])
        split_tensors = tf.split(split_tensor, self.attr_len, axis=-1)
        for i in range(self.attr_len):
            outputs[self.att_feats[i]] = tf.cast(tf.squeeze(split_tensors[i], axis=-1), tf.string)
        return outputs


class LookupTable(Layer):
    def __init__(self,
                 vocab_path: str = None,
                 default_value: int = 0,
                 **kwargs):
        self.vocab_path = vocab_path
        self.default_value = default_value

        if self.vocab_path:
            initializer = tf.lookup.TextFileInitializer(vocab_path,
                                                        'string',
                                                        1,
                                                        'int64',
                                                        0,
                                                        delimiter=',')
            self.table = tf.lookup.StaticHashTable(initializer,
                                                   default_value=self.default_value)

        super(LookupTable, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LookupTable, self).build(input_shape)

    def call(self, inputs):
        if not self.vocab_path:
            return inputs
        else:
            return self.table.lookup(inputs)

    def get_config(self):
        config = {
            'vocab_path': self.vocab_path,
            'default_value': self.default_value
        }
        base_config = super(LookupTable, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


class SequencePoolingLayer(Layer):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.
      Input shape
        - A list of two  tensor [seq_value,seq_len]
        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``
        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.
      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.
        - **supports_masking**:If True,the input need to support masking.
    """

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(SequencePoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(SequencePoolingLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            uiseq_embed_list = seq_value_len_list
            mask = tf.cast(mask, tf.float32)  # tf.to_float(mask)
            user_behavior_length = tf.reduce_sum(mask, axis=-1, keepdims=True)
            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list

            mask = tf.sequence_mask(user_behavior_length,
                                    self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = uiseq_embed_list.shape[-1]

        mask = tf.tile(mask, [1, 1, embedding_size])

        if self.mode == "max":
            hist = uiseq_embed_list - (1-mask) * 1e9
            return tf.reduce_max(hist, 1, keepdims=True)

        hist = tf.reduce_sum(uiseq_embed_list * mask, 1, keepdims=False)

        if self.mode == "mean":
            hist = tf.divide(hist, tf.cast(user_behavior_length, tf.float32) + self.eps)

        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return (None, 1, input_shape[-1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LabelAwareAttention(Layer):
    def __init__(self, k_max, pow_p=1, keepdims=True, **kwargs):
        self.k_max = k_max
        self.pow_p = pow_p
        self.keepdims = keepdims
        super(LabelAwareAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        self.embedding_size = input_shape[0][-1]
        super(LabelAwareAttention, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        keys = inputs[0]
        query = inputs[1]
        weight = tf.reduce_sum(keys * query, axis=-1, keepdims=True)
        weight = tf.pow(weight, self.pow_p)  # [x,k_max,1]

        if len(inputs) == 3:
            k_user = tf.cast(tf.maximum(
                1.,
                tf.minimum(
                    tf.cast(self.k_max, dtype="float32"),  # k_max
                    tf.math.log1p(tf.cast(inputs[2], dtype="float32")) / tf.math.log(2.)  # hist_len
                )
            ), dtype="int64")

            seq_mask = tf.transpose(tf.sequence_mask(k_user, self.k_max), [0, 2, 1])
            padding = tf.ones_like(seq_mask, dtype=tf.float32) * (-2 ** 32 + 1)  # [x,k_max,1]
            weight = tf.where(seq_mask, weight, padding)

        weight = tf.nn.softmax(weight, name="weight")
        output = tf.reduce_sum(keys * weight, axis=1, keepdims=self.keepdims)

        return output

    def compute_output_shape(self, input_shape):
        return (None, self.embedding_size)

    def get_config(self, ):
        config = {'k_max': self.k_max, 'pow_p': self.pow_p}
        base_config = super(LabelAwareAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CapsuleLayer(Layer):
    def __init__(self, input_units, out_units, max_len, k_max, iteration_times=3,
                 init_std=1.0, **kwargs):
        self.input_units = input_units
        self.out_units = out_units
        self.max_len = max_len
        self.k_max = k_max
        self.iteration_times = iteration_times
        self.init_std = init_std
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.routing_logits = self.add_weight(shape=[1, self.k_max, self.max_len],
                                              initializer=RandomNormal(stddev=self.init_std),
                                              trainable=False, name="B", dtype=tf.float32)
        self.bilinear_mapping_matrix = self.add_weight(shape=[self.input_units, self.out_units],
                                                       initializer=RandomNormal(
                                                           stddev=self.init_std),
                                                       name="S", dtype=tf.float32)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        behavior_embddings, seq_len = inputs
        batch_size = tf.shape(behavior_embddings)[0]
        seq_len_tile = tf.tile(seq_len, [1, self.k_max])

        for i in range(self.iteration_times):
            mask = tf.sequence_mask(seq_len_tile, self.max_len)
            pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 32 + 1)
            routing_logits_with_padding = tf.where(mask, tf.tile(
                self.routing_logits, [batch_size, 1, 1]), pad)
            weight = tf.nn.softmax(routing_logits_with_padding)
            behavior_embdding_mapping = tf.tensordot(
                behavior_embddings, self.bilinear_mapping_matrix, axes=1)
            Z = tf.matmul(weight, behavior_embdding_mapping)
            interest_capsules = squash(Z)

            delta_routing_logits = tf.reduce_sum(
                tf.matmul(interest_capsules, tf.transpose(
                    behavior_embdding_mapping, perm=[0, 2, 1])),
                axis=0, keepdims=True
            )
            self.routing_logits.assign_add(delta_routing_logits)

        interest_capsules = tf.reshape(interest_capsules, [-1, self.k_max, self.out_units])
        return interest_capsules

    def compute_output_shape(self, input_shape):
        return (None, self.k_max, self.out_units)

    def get_config(self, ):
        config = {'input_units': self.input_units, 'out_units': self.out_units, 'max_len': self.max_len,
                  'k_max': self.k_max, 'iteration_times': self.iteration_times, "init_std": self.init_std}
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def squash(inputs):
    vec_squared_norm = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-8)
    vec_squashed = scalar_factor * inputs

    return vec_squashed
