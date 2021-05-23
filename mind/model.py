import os
from typing import List, Tuple
import tensorflow as tf
from absl import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from core.base_model import BaseModel
from mind.layers import SequencePoolingLayer, CapsuleLayer, LabelAwareAttention
from mind.features import Feature, FeatureBuilder, SparseFeature


class MIND(BaseModel):
    """ref: Multi-interest network with dynamic routing for recommendation at Tmall
    """

    def __init__(self, features: Tuple[List[Feature]],
                 hist_max_len=10, num_sampled=4,
                 dynamic_k=False, k_max=3, p=1.0, user_dnn_hidden_units=(64, 32),
                 kernal_regularizer=None, model_name='mind_model'):
        if len(features) == 3:
            user_features, item_features, neg_item_features = features
        else:
            raise ValueError('features tuple is only three grams (user_features, item_features, neg_item_features).')
        if len(item_features) > 1 and len(neg_item_features) > 1:
            raise ValueError("Now MIND item_features only support 1 item feature like item_id, the same as neg_item_features.")
        self.feature_builder = FeatureBuilder(user_features + item_features + neg_item_features)
        self.item_feature = item_features[0]

        self.kernal_regularizer = kernal_regularizer
        self.k_max = k_max
        self.p = p
        self.dynamic_k = dynamic_k
        self.num_sampled = num_sampled
        self.hist_max_len = hist_max_len
        self.user_dnn_hidden_units = user_dnn_hidden_units
        self.model_name = model_name

        model = self._create_model(user_features, item_features, neg_item_features, hist_max_len, num_sampled, dynamic_k, k_max, p, user_dnn_hidden_units, model_name)
        super().__init__(model)

    def _create_model(self, user_features, item_features: List[SparseFeature], neg_item_features: List[SparseFeature], hist_max_len=10, num_sampled=4, dynamic_k=False, k_max=3, p=1.0, user_dnn_hidden_units=(64, 32), model_name='mind_model'):

        item_feature_name = item_features[0].name
        hist_item_feature_name = 'hist_' + item_feature_name
        hist_len_feature_name = 'hist_len'
        logging.info('NOTICE: user_features need have feature names: %s, %s', hist_item_feature_name, hist_len_feature_name)
        logging.info('user_features: %s, item_features: %s, neg_item_features: %s.', user_features, item_features, neg_item_features)
        item_embedding_dim = item_features[0].embedding_dim

        inputs = self.feature_builder.get_inputs()

        inputs_list = list(inputs.values())
        user_inputs_list = [inputs[uf.name] for uf in user_features]
        item_inputs_list = [inputs[item_feature_name]]

        user_embeds = self.feature_builder.lookup_embeddings(user_features)
        hist_item_id_emb = user_embeds.pop(hist_item_feature_name)
        hist_len = user_embeds.pop(hist_len_feature_name)

        item_embeds = self.feature_builder.lookup_embeddings(item_features)

        item_id_emb = item_embeds.pop(item_feature_name)

        pooling_hist_item_id = SequencePoolingLayer()([hist_item_id_emb, hist_len])

        user_feature_embeds = concatenate(list(user_embeds.values()) + [pooling_hist_item_id])
        # user_feature_embeds = concatenate(list(user_embeds.values()))
        user_feature_embeds = tf.tile(user_feature_embeds, [1, k_max, 1])

        user_high_capsule = CapsuleLayer(input_units=item_embedding_dim, out_units=item_embedding_dim,
                                         max_len=hist_max_len, k_max=k_max, init_std=5.0)([hist_item_id_emb, hist_len])

        user_deep_input = concatenate([user_feature_embeds, user_high_capsule])

        for i, u in enumerate(user_dnn_hidden_units):
            user_deep_input = Dense(u, activation="relu", kernel_regularizer=self.kernal_regularizer, name="FC_{0}".format(i+1))(user_deep_input)

        attention_layer = LabelAwareAttention(k_max=k_max, pow_p=p, keepdims=True)

        if dynamic_k:
            user_embedding_final = attention_layer([user_deep_input, item_id_emb, hist_len])
        else:
            user_embedding_final = attention_layer([user_deep_input, item_id_emb])

        # label
        label = tf.zeros_like(inputs[item_feature_name], dtype=tf.int32)

        neg_item_feature_name = neg_item_features[0].name
        neg_item_embeds = self.feature_builder.lookup_embeddings(neg_item_features)
        neg_item_id_emb = neg_item_embeds.pop(neg_item_feature_name)
        item_embedding_layer = concatenate([item_id_emb, neg_item_id_emb], axis=1)

        logits = tf.matmul(user_embedding_final, item_embedding_layer, transpose_b=True)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, logits)
        pred = tf.nn.softmax(logits)

        model = Model(inputs=inputs_list, outputs=pred, name=model_name)
        model.add_loss(tf.reduce_mean(loss))
        model.add_metric(tf.keras.metrics.sparse_categorical_accuracy(label, pred), name='acurracy')

        model.__setattr__("user_input", user_inputs_list)
        model.__setattr__("user_embedding", user_deep_input)

        model.__setattr__("item_input", item_inputs_list)
        model.__setattr__("item_embedding", item_id_emb)

        return model

    def get_user_model(self, normalized=True):
        model = self.model
        user_embedding = tf.nn.l2_normalize(model.user_embedding, axis=-1) if normalized else model.user_embedding
        user_embedding_model = Model(inputs=model.user_input, outputs=user_embedding, name='user_embedding_model')
        return user_embedding_model

    def get_item_model(self, normalized=True):
        model = self.model
        item_embedding = tf.nn.l2_normalize(model.item_embedding, axis=-1) if normalized else model.item_embedding
        item_embedding_model = Model(inputs=model.item_input, outputs=item_embedding, name='item_embedding_model')
        return item_embedding_model

    def get_serving_model(self):
        user_input = self.model.user_input
        norm_user_embedding = tf.nn.l2_normalize(self.model.user_embedding, axis=-1, name='norm_user')

        item_inputs = {'item_id': Input(shape=(None,), name="item_id", dtype=tf.string)}
        item_embeddings = self.feature_builder.get_item_embedding(item_inputs, self.item_feature)
        norm_item_embeddings = tf.nn.l2_normalize(item_embeddings, axis=-1, name='norm_item')

        probs = tf.einsum('ijk,ilk->ijl', norm_user_embedding, norm_item_embeddings)
        sorted_probs = tf.sort(probs, axis=-1, direction='DESCENDING')
        sorted_probs = tf.keras.layers.Lambda(lambda x: x, name='probs')(sorted_probs)
        sorted_index = tf.argsort(probs, axis=-1, direction='DESCENDING', stable=True)
        sorted_index = tf.keras.layers.Lambda(lambda x: x, name='indices')(sorted_index)
        serving_model = Model(inputs=user_input + list(item_inputs.values()), outputs={'probs': sorted_probs, 'indices': sorted_index}, name='serving_model')
        return serving_model
