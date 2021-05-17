import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, concatenate, Input
from mind.layers import MaskZero, ListMeanPooling, SplitString, LookupTable, SequencePoolingLayer, CapsuleLayer, LabelAwareAttention


class MIND(object):
    def __init__(self, vocab_paths, num_vocabs, embedding_dims, hist_max_len=10, num_sampled=4,
                 dynamic_k=False, k_max=3, p=1.0, user_dnn_hidden_units=(64, 32),
                 embeddings_regularizer=None, kernal_regularizer=None, model_name='mind_model'):

        user_feature_names = ['user_id', 'user_type', 'member_level']
        item_feature_names = ['product_id', 'first_class_id', 'second_class_id', 'third_class_id', 'brand_id']

        self.split_string_layers = {'item_id': SplitString(item_feature_names, name='split_item_id')}

        self.lookup_table_layers = {feature_name: LookupTable(vocab_paths[feature_name], name='lookup_'+feature_name) for feature_name in user_feature_names+item_feature_names}
        self.embeddings_regularizer = embeddings_regularizer
        self.kernal_regularizer = kernal_regularizer

        def gen_embedding(name):
            return Embedding(num_vocabs[name], embedding_dims[name], name='embedding_'+name, embeddings_initializer='glorot_normal', embeddings_regularizer=self.embeddings_regularizer, mask_zero=True)

        self.embedding_layers = {feature_name: gen_embedding(feature_name) for feature_name in user_feature_names+item_feature_names}

        self.model = self.create_model(embedding_dims['item_id'], hist_max_len, num_sampled, dynamic_k, k_max, p, user_dnn_hidden_units, model_name)

    def _lookup_embedding(self, inputs, name, mask_zero=False):
        ids = self.lookup_table_layers[name](inputs[name])
        embs = self.embedding_layers[name](ids)
        if mask_zero:
            return MaskZero(name='masked_emb_'+name)(embs)
        return embs

    def _get_item_embedding(self, inputs, name, combiner='mean'):
        split_inputs = self.split_string_layers['item_id'](inputs[name])
        embeds = {split_name: self._lookup_embedding(split_inputs, split_name) for split_name in split_inputs.keys()}
        if combiner == 'mean':
            embed = ListMeanPooling(name='mean_'+name)(embeds.values())
        elif combiner == 'concat':
            embed = concatenate(embeds.values(), name='concat_' + name)
        return embed

    def _get_embeddings(self, inputs, return_list=False):
        embeds = {}
        embeds['user_id'] = self._lookup_embedding(inputs, 'user_id', mask_zero=True)
        embeds['user_type'] = self._lookup_embedding(inputs, 'user_type', mask_zero=True)
        embeds['member_level'] = self._lookup_embedding(inputs, 'member_level', mask_zero=True)

        embeds['hist_item_id'] = self._get_item_embedding(inputs, 'hist_item_id')

        embeds['item_id'] = self._get_item_embedding(inputs, 'item_id')
        embeds['neg_item_id'] = self._get_item_embedding(inputs, 'neg_item_id')
        if return_list:
            return list(embeds.values())
        return embeds

    def create_model(self, item_embedding_dim, hist_max_len=10, num_sampled=4, dynamic_k=False, k_max=3, p=1.0, user_dnn_hidden_units=(64, 32), model_name='mind_model'):
        inputs = {
            # user_input
            'user_id': Input(shape=(1), name="user_id", dtype=tf.string),
            'user_type': Input(shape=(1), name="user_type", dtype=tf.string),
            'member_level': Input(shape=(1), name="member_level", dtype=tf.string),
            'hist_item_id': Input(shape=(hist_max_len), name="hist_item_id", dtype=tf.string),
            'hist_len': Input(shape=(1), name='hist_len', dtype=tf.int32),
            # item_input
            'item_id': Input(shape=(1), name="item_id", dtype=tf.string),
            # neg_sample_input
            'neg_item_id': Input(shape=(num_sampled), name="neg_item_id", dtype=tf.string),
        }
        inputs_list = list(inputs.values())
        user_inputs_list = [inputs['user_id'], inputs['user_type'],
                            inputs['member_level'], inputs['hist_item_id'], inputs['hist_len']]
        item_inputs_list = [inputs['item_id']]

        hist_len = inputs['hist_len']
        # label
        label = tf.zeros_like(inputs['item_id'], dtype=tf.int32)

        embeds = self._get_embeddings(inputs)

        hist_item_id_emb = embeds.pop('hist_item_id')
        item_id_emb = embeds.pop('item_id')
        neg_item_id_emb = embeds.pop('neg_item_id')

        pooling_hist_item_id = SequencePoolingLayer()([hist_item_id_emb, hist_len])

        # user_feature_embeds = concatenate(list(embeds.values()) + [pooling_hist_item_id])
        user_feature_embeds = concatenate(list(embeds.values()))
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

    def train(self, train_dataset, test_dataset, epochs, steps_per_epoch=None, checkpoint_path=None, checkpoint_frequency='epoch', restore_latest=False, monitor='val_loss', mode='min'):
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, verbose=1, patience=2, restore_best_weights=True)]
        if checkpoint_path is not None:
            if restore_latest:
                self.load_weights(checkpoint_path, restore_latest)
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_freq=checkpoint_frequency, monitor=monitor, mode=mode,
                                                               verbose=1, save_weights_only=True, save_best_only=True)
            callbacks.append(checkpoint_cb)

        history = self.model.fit(train_dataset,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=test_dataset,
                                 validation_steps=100,
                                 callbacks=callbacks)
        return history

    def compile(self, optimizer="adam"):
        self.model.compile(optimizer=optimizer)

    def save_model(self, saved_model_path, version=None, model=None):
        model = model if model else self.model
        if version:
            saved_model_path = saved_model_path + "/" + version
        model.save(saved_model_path, signatures=model.call)

    def load_weights(self, checkpoint_path, latest=False):
        if latest:
            checkpoint_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
            latest_path = tf.train.latest_checkpoint(checkpoint_dir)
            print('latest checkpoint dir: ', latest_path)
            self.model.load_weights(latest_path)
        else:
            self.model.load_weights(checkpoint_path)

    def save_weights(self, checkpoint_path):
        self.model.save_weights(checkpoint_path)

    def get_model(self):
        return self.model

    def summary(self):
        self.model.summary()

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

        item_inputs = {'item_ids': Input(shape=(None,), name="item_ids", dtype=tf.string)}
        item_embeddings = self._get_item_embedding(item_inputs, 'item_ids')
        norm_item_embeddings = tf.nn.l2_normalize(item_embeddings, axis=-1, name='norm_item')

        probs = tf.einsum('ijk,ilk->ijl', norm_user_embedding, norm_item_embeddings, name='probs')
        sorted_probs = tf.sort(probs, axis=-1, direction='DESCENDING')
        sorted_index = tf.argsort(probs, axis=-1, direction='DESCENDING', stable=True)
        serving_model = Model(inputs=user_input + list(item_inputs.values()), outputs=(sorted_probs, sorted_index), name='serving_model')
        return serving_model

    def save_serving_model(self, saved_model_path, version=None):
        serving_model = self.get_serving_model()
        self.save_model(saved_model_path, version=version, model=serving_model)
