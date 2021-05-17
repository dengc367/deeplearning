#!/usr/bin/env python
# coding: utf-8

# In[2]:


import subprocess
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import pickle


# In[3]:


prefix = 'word2vec'

train_path = "data/train.tfrecord"
test_path = "data/test.tfrecord"


def shellCmd(cmd):
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE
                            ).stdout.readline().decode("utf8").replace("\n", "")


HDFS_URL = shellCmd("hdfs getconf -confKey fs.defaultFS")
HDFS_WORK_DIRECTORY = HDFS_URL+"/user/"+os.environ["USER"]
train_path = HDFS_WORK_DIRECTORY+"/" + prefix + "/"+train_path
test_path = HDFS_WORK_DIRECTORY + "/" + prefix + "/"+test_path


# In[6]:


meta_path = "meta/data_meta.pkl"
model_meta_path = "meta/model_meta.pkl"
checkpoint_dir = 'model/saved_ckpt_path'
checkpoint_path = checkpoint_dir + '/cp-{epoch:04d}.ckpt'
# checkpoint_path = checkpoint_dir + '/cp-0.ckpt'
checkpoint_frequency = 2000
# checkpoint_frequency='epoch'

buffer_size = 1024
batch_size = 128
num_epochs = 4
learning_rate = 0.001


embedding_size = {
    'item_id': 100,
    'product_id': 100,
    'store_id': 10,
    'brand_id': 20,
    'first_class_id': 10,
    'second_class_id': 10,
    'third_class_id': 20
}

# embedding_size = {
#    'item_id': 50,
#    'product_id': 50,
#    'store_id': 5,
#    'brand_id': 10,
#    'first_class_id': 5,
#    'second_class_id': 5,
#    'third_class_id': 10
# }


with open(meta_path, 'rb') as f:
    vocab_paths = pickle.load(f)
    print('vocab_paths: ', vocab_paths)
    num_vocab = pickle.load(f)
    print('num_vocab: ', num_vocab)
    context_length, neg_sample_num = pickle.load(f)
    print('context_length: ', context_length, ', neg_sample_num: ', neg_sample_num)
#     session_size, train_size, test_size = pickle.load(f)
#     print('session_size: ', session_size, 'train_size: ', train_size, 'test_size: ', test_size)

with open(model_meta_path, 'wb') as f:
    pickle.dump(embedding_size, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump((buffer_size, batch_size, neg_sample_num, num_epochs, learning_rate), f, protocol=pickle.HIGHEST_PROTOCOL)

# In[4]:


class SplitStringLayer(layers.Layer):
    """
    Keras Model
    TODO 还需要完善
    """

    def __init__(self, sep='_', **kwargs):
        self.sep = sep
        super(SplitStringLayer, self).__init__(**kwargs)

    def call(self, inputs):
        outputs = {}
        splits = tf.strings.split(inputs, sep=self.sep).to_tensor()
        outputs['item_id'] = inputs
        outputs['product_id'] = splits[..., 0]
        outputs['store_id'] = splits[..., 1]
        outputs['first_class_id'] = splits[..., 2]
        outputs['second_class_id'] = tf.cast(splits[..., 3], tf.string)
        outputs['third_class_id'] = tf.cast(splits[..., 4], tf.string)
        outputs['brand_id'] = tf.cast(splits[..., 5], tf.string)
        return outputs


class VocabLookupLayer(layers.Layer):
    def __init__(self, df_series_or_filename, default_value=-1, **kwargs):
        self.df_series_or_filename = df_series_or_filename
        if isinstance(self.df_series_or_filename, str):
            initializer = tf.lookup.TextFileInitializer(
                self.df_series_or_filename,
                tf.string,
                1,
                tf.int64,
                0, delimiter=',')
        else:
            raise Exception('support str or pd.core.series.Series types only!')
        self.table = tf.lookup.StaticHashTable(initializer, default_value=default_value)
        super(VocabLookupLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return self.table.lookup(inputs)


class CustomEmbedding(layers.Layer):
    def __init__(self, input_dim, output_dim, default_value=-1, mask_zero=False, embeddings_initializer='glorot_normal', embeddings_regularizer=None, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero
        self.default_value = default_value
        self.embeddings = layers.Embedding(self.input_dim, self.output_dim, embeddings_initializer=embeddings_initializer, name=self.name+'_embedding')

    def call(self, inputs):
        mask = tf.cast(tf.not_equal(inputs, self.default_value), tf.int64)
        outputs = self.embeddings(inputs * mask)
        tiled_shape = tuple(np.ones(inputs.shape.rank, dtype=int)) + (self.output_dim,)
        tiled_mask = tf.cast(tf.tile(tf.expand_dims(mask, -1), tiled_shape), tf.float32)
        outputs = outputs * tiled_mask
        return outputs

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, 0)


class Word2VecKerasModel(keras.Model):
    def __init__(self, vocab_paths, num_vocab, embedding_size, alpha=1, beta=0.01, num_sampled=2, learning_rate=0.01, **kvargs):
        self.alpha = alpha
        self.beta = beta
        self.num_sampled = num_sampled
        self.num_classes = num_vocab['item_id']
        self.total_embedding_size = np.sum(list(embedding_size.values()))
        self.learning_rate = learning_rate

        super(Word2VecKerasModel, self).__init__(**kvargs)

#         self.embedings_regularizer=keras.regularizers.L2(l2=self.beta)
        self.embedings_regularizer = None
        self.l2_regularizer = keras.regularizers.L2(l2=self.beta)

        self.embedding_params = {
            'item_id': CustomEmbedding(num_vocab['item_id'], embedding_size['item_id'], name='item_id_param'),
            'product_id': CustomEmbedding(num_vocab['product_id'], embedding_size['product_id'], name='product_id_param'),
            'store_id': CustomEmbedding(num_vocab['store_id'], embedding_size['store_id'], name='store_id_param'),
            'brand_id': CustomEmbedding(num_vocab['brand_id'], embedding_size['brand_id'], name='brand_id_param'),
            'first_class_id': CustomEmbedding(num_vocab['first_class_id'], embedding_size['first_class_id'], name='first_class_id_param'),
            'second_class_id': CustomEmbedding(num_vocab['second_class_id'], embedding_size['second_class_id'], name='second_class_id_param'),
            'third_class_id': CustomEmbedding(num_vocab['third_class_id'], embedding_size['third_class_id'], name='third_class_id_param')
        }

        self.weight_params = {
            'item_id': CustomEmbedding(num_vocab['item_id'], embedding_size['item_id'], name='context_item_id_param', embeddings_regularizer=self.embedings_regularizer),
            'product_id': CustomEmbedding(num_vocab['product_id'], embedding_size['product_id'], name='context_product_id_param', embeddings_regularizer=self.embedings_regularizer),
            'store_id': CustomEmbedding(num_vocab['store_id'], embedding_size['store_id'], name='context_store_id_param', embeddings_regularizer=self.embedings_regularizer),
            'brand_id': CustomEmbedding(num_vocab['brand_id'], embedding_size['brand_id'], name='context_brand_id_param', embeddings_regularizer=self.embedings_regularizer),
            'first_class_id': CustomEmbedding(num_vocab['first_class_id'], embedding_size['first_class_id'], name='context_first_class_id_param', embeddings_regularizer=self.embedings_regularizer),
            'second_class_id': CustomEmbedding(num_vocab['second_class_id'], embedding_size['second_class_id'], name='context_second_class_id_param', embeddings_regularizer=self.embedings_regularizer),
            'third_class_id': CustomEmbedding(num_vocab['third_class_id'], embedding_size['third_class_id'], name='context_third_class_id_param', embeddings_regularizer=self.embedings_regularizer)
        }

        self.structural_weights = {
            # [m1, m2]
            'product_id': self.add_weight(shape=(embedding_size['item_id'], embedding_size['product_id']), name='product_id_attr', initializer="glorot_normal", regularizer=self.l2_regularizer),
            'store_id': self.add_weight(shape=(embedding_size['item_id'], embedding_size['store_id']), name='store_id_attr', initializer="glorot_normal", regularizer=self.l2_regularizer),
            'brand_id': self.add_weight(shape=(embedding_size['item_id'], embedding_size['brand_id']), name='brand_id_attr', initializer="glorot_normal", regularizer=self.l2_regularizer),
            'first_class_id': self.add_weight(shape=(embedding_size['item_id'], embedding_size['first_class_id']), name='first_class_id_attr', initializer="glorot_normal", regularizer=self.l2_regularizer),
            'second_class_id': self.add_weight(shape=(embedding_size['item_id'], embedding_size['second_class_id']), name='second_class_id_attr', initializer="glorot_normal", regularizer=self.l2_regularizer),
            'third_class_id': self.add_weight(shape=(embedding_size['item_id'], embedding_size['third_class_id']), name='third_class_id_attr', initializer="glorot_normal", regularizer=self.l2_regularizer)
        }

        self.structural_weights = {
            'product_id': self.add_weight(shape=(embedding_size['item_id'], embedding_size['product_id']), name='product_id_attr', initializer="glorot_normal"),  # [m1, m2]
            'store_id': self.add_weight(shape=(embedding_size['item_id'], embedding_size['store_id']), name='store_id_attr', initializer="glorot_normal"),
            'brand_id': self.add_weight(shape=(embedding_size['item_id'], embedding_size['brand_id']), name='brand_id_attr', initializer="glorot_normal"),
            'first_class_id': self.add_weight(shape=(embedding_size['item_id'], embedding_size['first_class_id']), name='first_class_id_attr', initializer="glorot_normal"),
            'second_class_id': self.add_weight(shape=(embedding_size['item_id'], embedding_size['second_class_id']), name='second_class_id_attr', initializer="glorot_normal"),
            'third_class_id': self.add_weight(shape=(embedding_size['item_id'], embedding_size['third_class_id']), name='third_class_id_attr', initializer="glorot_normal")
        }
        self.split_string_layer = SplitStringLayer(name='split_string')
        self.vocab_lookup_layer = {
            'item_id': VocabLookupLayer(vocab_paths['item_id'], name='item_id_lookup'),
            'product_id': VocabLookupLayer(vocab_paths['product_id'], name='product_id_lookup'),
            'brand_id': VocabLookupLayer(vocab_paths['brand_id'], name='brand_id_lookup'),
            'store_id': VocabLookupLayer(vocab_paths['store_id'], name='store_id_lookup'),
            'first_class_id': VocabLookupLayer(vocab_paths['first_class_id'], name='first_class_id_lookup'),
            'second_class_id': VocabLookupLayer(vocab_paths['second_class_id'], name='second_class_id_lookup'),
            'third_class_id': VocabLookupLayer(vocab_paths['third_class_id'], name='third_class_id_lookup')
        }

    def _lookup_ids(self, splits):
        indices = {}
        indices['item_id'] = self.vocab_lookup_layer.get('item_id')(splits['item_id'])
        indices['product_id'] = self.vocab_lookup_layer.get('product_id')(splits['product_id'])
        indices['store_id'] = self.vocab_lookup_layer.get('store_id')(splits['store_id'])
        indices['brand_id'] = self.vocab_lookup_layer.get('brand_id')(splits['brand_id'])
        indices['first_class_id'] = self.vocab_lookup_layer.get('first_class_id')(splits['first_class_id'])
        indices['second_class_id'] = self.vocab_lookup_layer.get('second_class_id')(splits['second_class_id'])
        indices['third_class_id'] = self.vocab_lookup_layer.get('third_class_id')(splits['third_class_id'])
        return indices

    def lookup_ids(self, inputs_x, inputs_y):
        splits_x = self.split_string_layer(inputs_x)
        inputs_indices = self._lookup_ids(splits_x)

        splits_y = self.split_string_layer(inputs_y)
        labels_indices = self._lookup_ids(splits_y)

        return inputs_indices, labels_indices

    def get_target_embeddings(self, ids):
        embeddings = {}
        embeddings['item_id'] = self.embedding_params['item_id'](ids['item_id'])  # [N, m1]
        embeddings['product_id'] = self.embedding_params['product_id'](ids['product_id'])  # [N, m2]
        embeddings['store_id'] = self.embedding_params['store_id'](ids['store_id'])  # [N, m3]
        embeddings['brand_id'] = self.embedding_params['brand_id'](ids['brand_id'])  # [N, m4]
        embeddings['first_class_id'] = self.embedding_params['first_class_id'](ids['first_class_id'])
        embeddings['second_class_id'] = self.embedding_params['second_class_id'](ids['second_class_id'])
        embeddings['third_class_id'] = self.embedding_params['third_class_id'](ids['third_class_id'])
        return embeddings

    def get_context_weights(self, ids):
        embeddings = {}
        embeddings['item_id'] = self.weight_params['item_id'](ids['item_id'])  # [N, m1]
        embeddings['product_id'] = self.weight_params['product_id'](ids['product_id'])  # [N, m2]
        embeddings['store_id'] = self.weight_params['store_id'](ids['store_id'])  # [N, m3]
        embeddings['brand_id'] = self.weight_params['brand_id'](ids['brand_id'])  # [N, m4]
        embeddings['first_class_id'] = self.weight_params['first_class_id'](ids['first_class_id'])
        embeddings['second_class_id'] = self.weight_params['second_class_id'](ids['second_class_id'])
        embeddings['third_class_id'] = self.weight_params['third_class_id'](ids['third_class_id'])
        return embeddings

    def get_embeddings(self, input_ids, labels_ids):

        inputs_embeddings = self.get_target_embeddings(input_ids)
        labels_embeddings = self.get_context_weights(labels_ids)

        return inputs_embeddings, labels_embeddings

    def structural_loss(self, embeddings):  # [N, m1] @ [m1, m2+m3+...+m7] * [N, m2+m3+...+m7] = [N, m2+m3+...+m7]
        item_embeddings = embeddings.pop('item_id')
        structural_weights = tf.concat(list(self.structural_weights.values()), axis=-1)  # [m1, m2+m3+...+m7]
        attribute_embeddings = tf.concat(list(embeddings.values()), axis=-1)  # [N, m2+m3+...+m7]

        loss = - self.alpha * tf.reduce_mean(tf.math.log_sigmoid(tf.reduce_sum(item_embeddings @ structural_weights * attribute_embeddings, axis=-1)))
        # penalty = self.l2_regularizer(structural_weights)
        # return loss + penalty
        return loss

    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""
        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)
        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.

        loss = tf.reduce_mean(true_xent + tf.reduce_sum(sampled_xent, -1))  # / true_logits.shape[0]
        return loss

    def call(self, inputs):
        inputs_x, inputs_y = inputs['target'], inputs['context']  # [N ,1], , [N, 1+n]

        target_ids, context_ids = self.lookup_ids(inputs_x, inputs_y)  # [N, 1], [N, 1+n]
        target_embeddings, context_embeddings = self.get_embeddings(target_ids, context_ids)  # [N, 1, m1+...+m7], [N, 1+n, m1+...+m7]
        concated_target_embeddings = tf.concat(list(target_embeddings.values()), axis=-1)  # [N, 1, m1+...+m7]
        context_weights = tf.concat(list(context_embeddings.values()), axis=-1)  # [N, 1+n, m1+...+m7]

        logits = tf.einsum('ijk,ikl->il', concated_target_embeddings, tf.transpose(context_weights, (0, 2, 1)))

        nce_loss = self.nce_loss(logits[..., 0], logits[..., 1:])
        structural_loss = self.structural_loss(target_embeddings)
        loss = nce_loss + structural_loss

        self.add_loss(loss)
        return loss

# In[5]:


def decode(serialized):
    example = tf.io.parse_single_example(serialized,
                                         features={
                                             'target': tf.io.FixedLenFeature([1], tf.string),
                                             'context': tf.io.FixedLenFeature([1 + neg_sample_num], tf.string)
                                         })
    return example


train_file_pattern = tf.data.Dataset.list_files(train_path + "/part-*")
test_file_pattern = tf.data.Dataset.list_files(test_path + "/part-*")
train_file_pattern = train_file_pattern.concatenate(test_file_pattern)
train_dataset = tf.data.TFRecordDataset(train_file_pattern).map(decode).prefetch(buffer_size*10).shuffle(buffer_size).batch(batch_size)
test_dataset = tf.data.TFRecordDataset(test_file_pattern).map(decode).prefetch(buffer_size*10).shuffle(buffer_size).batch(batch_size)


# In[6]:

word2vec = Word2VecKerasModel(vocab_paths, num_vocab, embedding_size, num_sampled=neg_sample_num, name='word2vec_keras_model')
schedule_learning_rate = keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps=10000, decay_rate=0.8, staircase=False)
word2vec.compile(keras.optimizers.Adam(schedule_learning_rate), None)

checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    #     save_best_only=True,
    save_freq=checkpoint_frequency,
    #     validation_split=0.2
)

word2vec.fit(x=train_dataset, y=None, epochs=num_epochs, validation_data=test_dataset, callbacks=[cp_callback])
