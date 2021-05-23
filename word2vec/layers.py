import tensorflow as tf

from tensorflow.keras.layers import Layer
# from tensorflow.keras.initializers import RandomNormal


class NCE(Layer):
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


class StructuralConnenctions(Layer):
    """structural loss
    """

    def __init__(self, item_feature_name, attr_feature_names, embedding_dims, l2_reg=None, **kwargs) -> None:
        super().__init__(name='structural_loss', **kwargs)
        self.item_feature_name = item_feature_name

        self.structural_weights = {name: self.add_weight(shape=(embedding_dims[item_feature_name], embedding_dims[name]), name='structural_' + name,
                                                         initializer="glorot_normal", regularizer=l2_reg) for name in attr_feature_names}

    def build(self, input_shape):
        pass

    def call(self, inputs, mask=None, training=True):  # [N, m1] @ [m1, m2+m3+...+m7] * [N, m2+m3+...+m7] = [N, m2+m3+...+m7]
        self.training = training
        embeddings = inputs
        item_embeddings = inputs.pop(self.item_feature_name)
        structural_weights = tf.concat(list(self.structural_weights.values()), axis=-1)  # [m1, m2+m3+...+m7]
        attribute_embeddings = tf.concat(list(embeddings.values()), axis=-1)  # [N, m2+m3+...+m7]

        if training:
            loss = - tf.reduce_mean(tf.math.log_sigmoid(tf.reduce_sum(item_embeddings @ structural_weights * attribute_embeddings, axis=-1)))
            return loss

        # when the value is masked, the mask value is False
        item_masked = mask[self.item_feature_name]
        computed_item_embeddings = tf.matmul(attribute_embeddings, structural_weights, transpose_b=True)  # [N, n, m1]
        cond_item_embeddings = tf.where(item_masked, item_embeddings, computed_item_embeddings)

        embeddings[self.item_feature_name] = cond_item_embeddings
        return embeddings
