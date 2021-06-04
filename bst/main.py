# %%
import tensorflow as tf
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
from bst.model import BST
import pickle
from absl import logging

prefix = "mind"
train_path = prefix + "/data/train.tfrecord"
test_path = prefix + "/data/train.tfrecord"
meta_path = prefix + "/meta/data_meta.pkl"
model_meta_path = prefix + "/meta/model_meta.pkl"
checkpoint_dir = prefix + "/model/saved_ckpt_path"
checkpoint_path = checkpoint_dir + '/cp-{epoch:04d}.ckpt'
checkpoint_frequency = 'epoch'
epochs = 2
buffer_size = 1024
batch_size = 64
test_batch_size = 1
embedding_dim = 32

with open(meta_path, 'rb') as f:
    vocab_paths = pickle.load(f)
    num_vocabs = pickle.load(f)
    context_length, neg_sample_num = pickle.load(f)

vocab_paths = dict([(k, prefix + '/' + v) for k, v in vocab_paths.items()])
vocab_paths, num_vocabs, context_length, neg_sample_num
# %%

embedding_dims = {
    'user_id': embedding_dim,
    'user_type': 8,
    'member_level': 8,
    'item_id': embedding_dim,
    'product_id': embedding_dim,
    'first_class_id': embedding_dim,
    'second_class_id': embedding_dim,
    'third_class_id': embedding_dim,
    'brand_id': embedding_dim
}
seq_length = context_length

user_profile_feature_columns = [
    SparseFeat('user_id',
               num_vocabs['user_id'],
               dtype='string',
               use_hash=True,
               vocabulary_path=vocab_paths['user_id'],
               embedding_dim=embedding_dims['user_id']),
    SparseFeat('user_type',
               num_vocabs['user_type'],
               dtype='string',
               use_hash=True,
               vocabulary_path=vocab_paths['user_type'],
               embedding_dim=embedding_dims['user_type']),
    SparseFeat('member_level',
               num_vocabs['member_level'],
               dtype='string',
               use_hash=True,
               vocabulary_path=vocab_paths['member_level'],
               embedding_dim=embedding_dims['member_level']),
]

item_profile_feature_columns = [
    SparseFeat('product_id',
               num_vocabs['product_id'],
               dtype='string',
               use_hash=True,
               vocabulary_path=vocab_paths['product_id'],
               embedding_dim=embedding_dims['product_id'],
               embedding_name='product_id'),
    SparseFeat('first_class_id',
               num_vocabs['first_class_id'],
               dtype='string',
               use_hash=True,
               vocabulary_path=vocab_paths['first_class_id'],
               embedding_dim=embedding_dims['first_class_id'],
               embedding_name='first_class_id'),
    SparseFeat('second_class_id',
               num_vocabs['second_class_id'],
               dtype='string',
               use_hash=True,
               vocabulary_path=vocab_paths['second_class_id'],
               embedding_dim=embedding_dims['second_class_id'],
               embedding_name='second_class_id'),
    SparseFeat('third_class_id',
               num_vocabs['third_class_id'],
               dtype='string',
               use_hash=True,
               vocabulary_path=vocab_paths['third_class_id'],
               embedding_dim=embedding_dims['third_class_id'],
               embedding_name='third_class_id'),
    SparseFeat('brand_id',
               num_vocabs['brand_id'],
               dtype='string',
               use_hash=True,
               vocabulary_path=vocab_paths['brand_id'],
               embedding_dim=embedding_dims['brand_id'],
               embedding_name='brand_id')
]

hist_item_profile_feature_columns = [
    VarLenSparseFeat(SparseFeat('hist_product_id',
                                num_vocabs['product_id'],
                                dtype='string',
                                use_hash=True,
                                vocabulary_path=vocab_paths['product_id'],
                                embedding_dim=embedding_dims['product_id'],
                                embedding_name='product_id'),
                     maxlen=seq_length,
                     length_name="seq_length"),
    VarLenSparseFeat(SparseFeat('hist_first_class_id',
                                num_vocabs['first_class_id'],
                                dtype='string',
                                use_hash=True,
                                vocabulary_path=vocab_paths['first_class_id'],
                                embedding_dim=embedding_dims['first_class_id'],
                                embedding_name='first_class_id'),
                     maxlen=seq_length,
                     length_name="seq_length"),
    VarLenSparseFeat(SparseFeat(
        'hist_second_class_id',
        num_vocabs['second_class_id'],
        dtype='string',
        use_hash=True,
        vocabulary_path=vocab_paths['second_class_id'],
        embedding_dim=embedding_dims['second_class_id'],
        embedding_name='second_class_id'),
        maxlen=seq_length,
        length_name="seq_length"),
    VarLenSparseFeat(SparseFeat('hist_third_class_id',
                                num_vocabs['third_class_id'],
                                dtype='string',
                                use_hash=True,
                                vocabulary_path=vocab_paths['third_class_id'],
                                embedding_dim=embedding_dims['third_class_id'],
                                embedding_name='third_class_id'),
                     maxlen=seq_length,
                     length_name="seq_length"),
    VarLenSparseFeat(SparseFeat('hist_brand_id',
                                num_vocabs['brand_id'],
                                dtype='string',
                                use_hash=True,
                                vocabulary_path=vocab_paths['brand_id'],
                                embedding_dim=embedding_dims['brand_id'],
                                embedding_name='brand_id'),
                     maxlen=seq_length,
                     length_name="seq_length"),
]
context_feature_columns = [
    # DenseFeat('pay_score', 1)
]
feature_columns = user_profile_feature_columns + item_profile_feature_columns + hist_item_profile_feature_columns + context_feature_columns
behavior_feature_list = [
    'product_id', 'first_class_id', 'second_class_id', 'third_class_id',
    'brand_id'
]

# %%
model = BST(dnn_feature_columns=feature_columns,
            history_feature_list=behavior_feature_list,
            att_head_num=4)
model.summary()

# %%


def gen_dataset(input_path, batch):
    def decode(serialized):
        example = tf.io.parse_single_example(
            serialized,
            features={
                'user_id':
                tf.io.FixedLenFeature([1], tf.string),
                'user_type':
                tf.io.FixedLenFeature([1], tf.string),
                'member_level':
                tf.io.FixedLenFeature([1], tf.string),
                'hist_ids':
                tf.io.FixedLenFeature([context_length], tf.string),
                'hist_len':
                tf.io.FixedLenFeature([1], tf.int64),
                'label_ids':
                tf.io.FixedLenFeature([1 + neg_sample_num], tf.string),
            })
        # example['hist_item_id'] = example.pop('hist_ids')

        hist_item_id = example.pop('hist_ids')  # [H]
        hist_item_id = tf.strings.split(hist_item_id, sep='_',
                                        maxsplit=5).to_tensor(
                                            shape=[context_length,
                                                   5])  # [H, 5]
        hist_item_id = tf.split(hist_item_id, 5, axis=-1)
        example['hist_product_id'] = tf.squeeze(hist_item_id[0])
        example['hist_first_class_id'] = tf.squeeze(hist_item_id[1])
        example['hist_second_class_id'] = tf.squeeze(hist_item_id[2])
        example['hist_third_class_id'] = tf.squeeze(hist_item_id[3])
        example['hist_brand_id'] = tf.squeeze(hist_item_id[4])

        # example['hist_len'] = tf.cast(example['hist_len'], tf.int32)
        example['seq_length'] = tf.cast(example.pop('hist_len'), tf.int32)

        # example['item_id'], example['neg_item_id'] = tf.split(
        #     example.pop('label_ids'), [1, neg_sample_num], axis=-1)

        label_ids = example.pop('label_ids')  # [1+N]

        split_label_ids = tf.strings.split(label_ids, sep='_',
                                           maxsplit=5).to_tensor(
                                               shape=[1 + neg_sample_num,
                                                      5])  # [1+N, 5]

        example['product_id'], example['first_class_id'], example[
            'second_class_id'], example['third_class_id'], example[
                'brand_id'] = tf.split(split_label_ids, 5, axis=-1)
        return example

    def flat_map(e):
        n = 1 + neg_sample_num
        k = {
            'user_id': [e['user_id']] * n,
            'user_type': [e['user_type']] * n,
            'member_level': [e['member_level']] * n,
            'hist_product_id': [e['hist_product_id']] * n,
            'hist_first_class_id': [e['hist_first_class_id']] * n,
            'hist_second_class_id': [e['hist_second_class_id']] * n,
            'hist_third_class_id': [e['hist_third_class_id']] * n,
            'hist_brand_id': [e['hist_brand_id']] * n,
            'seq_length': [e['seq_length']] * n,
            'product_id':
            e['product_id'],
            'first_class_id':
            e['first_class_id'],
            'second_class_id':
            e['second_class_id'],
            'third_class_id':
            e['third_class_id'],
            'brand_id':
            e['brand_id'],
            'y':
            tf.convert_to_tensor([[1]] + [[0]] * neg_sample_num,
                                 dtype=tf.int32)
        }
        return tf.data.Dataset.from_tensor_slices(k)

    def split_x_y(e):
        x = e
        y = x.pop('y')
        return (x, y)

    input_file_pattern = tf.data.Dataset.list_files(input_path)
    dataset = tf.data.TFRecordDataset(input_file_pattern, buffer_size=10000, num_parallel_reads=4).map(decode)
    dataset = dataset.flat_map(flat_map)
    dataset = dataset.map(split_x_y)
    dataset = dataset.prefetch(10000).shuffle(10000)
    dataset = dataset.batch(batch)
    return dataset


train_dataset = gen_dataset(train_path, batch_size)
# next(train_dataset.as_numpy_iterator())
# next(iter(train_dataset))

# %%

# %%

model.compile(optimizer=tf.optimizers.Adam(0.01), loss='binary_crossentropy', metrics=['binary_crossentropy'])
model.train(train_dataset, None, epochs=epochs)

# %%
