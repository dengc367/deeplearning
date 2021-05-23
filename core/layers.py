import tensorflow as tf
from tensorflow.keras.layers import Layer


class SplitString(Layer):
    """
    String Splitter Layer
    """

    def __init__(self,
                 attr_feats,
                 sep='_',
                 feature_name=None,
                 **kwargs):
        self.attr_feats = attr_feats
        self.attr_len = len(attr_feats)
        self.sep = sep
        self.feature_name = feature_name
        super().__init__(**kwargs)

    def call(self, inputs):
        outputs = {}
        if self.feature_name is not None:
            outputs[self.feature_name] = inputs
        if self.attr_len <= 1:
            return outputs
        split_tensor = tf.strings.split(inputs, sep=self.sep).to_tensor(shape=inputs.shape + [self.attr_len])
        split_tensors = tf.split(split_tensor, self.attr_len, axis=-1)
        for i in range(self.attr_len):
            outputs[self.attr_feats[i]] = tf.cast(tf.squeeze(split_tensors[i], axis=-1), tf.string)
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
        base_config = super().get_config()
        return dict(list(config.items()) + list(base_config.items()))


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
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
