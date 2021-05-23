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
        base_config = super(LookupTable, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


class NCELoss(Layer):
    """Noise Constrastive Loss
    """

    def __init__(self, num_true, num_sampled, **kwargs):
        super().__init__(name='nce_loss', **kwargs)
        self.num_true = num_true
        self.num_sampled = num_sampled

    def call(self, inputs):
        """Build the graph for the NCE loss."""
        target_embeddings, context_weights = inputs
        target_embeddings = tf.concat(list(target_embeddings.values()), axis=-1)  # [N, 1, m1+...+m7]
        context_weights = tf.concat(list(context_weights.values()), axis=-1)  # [N, 1+n, m1+...+m7]

        logits = tf.einsum('ijk,ikl->il', target_embeddings, tf.transpose(context_weights, (0, 2, 1)))

        true_logits, sampled_logits = tf.split(logits, [self.num_true, self.num_sampled], axis=-1)
        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)
        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.

        loss = tf.reduce_mean(true_xent + tf.reduce_sum(sampled_xent, -1))  # / true_logits.shape[0]
        return loss


class StructuralLoss(Layer):
    """structural loss
    """

    def __init__(self, item_feature_name, attr_feature_names, embedding_dims, l2_reg=None, **kwargs) -> None:
        super().__init__(name='structural_loss', **kwargs)
        self.item_feature_name = item_feature_name

        self.structural_weights = {name: self.add_weight(shape=(embedding_dims[item_feature_name], embedding_dims[name]), name='structural_' + name,
                                                         initializer="glorot_normal", regularizer=l2_reg) for name in attr_feature_names}

    def call(self, inputs):  # [N, m1] @ [m1, m2+m3+...+m7] * [N, m2+m3+...+m7] = [N, m2+m3+...+m7]
        embeddings = inputs
        item_embeddings = inputs.pop(self.item_feature_name)
        structural_weights = tf.concat(list(self.structural_weights.values()), axis=-1)  # [m1, m2+m3+...+m7]
        attribute_embeddings = tf.concat(list(embeddings.values()), axis=-1)  # [N, m2+m3+...+m7]

        loss = - tf.reduce_mean(tf.math.log_sigmoid(tf.reduce_sum(item_embeddings @ structural_weights * attribute_embeddings, axis=-1)))
        return loss
