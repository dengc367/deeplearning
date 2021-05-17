# import pytest
import tensorflow as tf

import mind.features as mf

# PYTHONPATH=. pytest -q test/test_features.py

vocab_paths = {'item_id': 'dev/meta/item_ids.csv', 'product_id': 'dev/meta/product_ids.csv', 'first_class_id': 'dev/meta/first_class_ids.csv', 'second_class_id': 'dev/meta/second_class_ids.csv',
               'third_class_id': 'dev/meta/third_class_ids.csv', 'brand_id': 'dev/meta/brand_ids.csv', 'user_id': 'dev/meta/user_ids.csv', 'user_type': 'dev/meta/user_types.csv', 'member_level': 'dev/meta/member_levels.csv'}
num_vocab = {'item_id': 5760, 'product_id': 5760, 'first_class_id': 14, 'second_class_id': 94, 'third_class_id': 450, 'brand_id': 5760, 'user_id': 56537, 'user_type': 15, 'member_level': 6}


item_id_attribute_features = [
    mf.SparseFeature(name='product_id', lookup_table_path=vocab_paths['product_id'],  embedding_intput_dim=num_vocab['product_id']),
    mf.SparseFeature(name='first_class_id', lookup_table_path=vocab_paths['first_class_id'], embedding_intput_dim=num_vocab['first_class_id']),
    mf.SparseFeature(name='second_class_id', lookup_table_path=vocab_paths['second_class_id'],  embedding_intput_dim=num_vocab['second_class_id']),
    mf.SparseFeature(name='third_class_id', lookup_table_path=vocab_paths['third_class_id'], embedding_intput_dim=num_vocab['third_class_id']),
    mf.SparseFeature(name='brand_id', lookup_table_path=vocab_paths['brand_id'], embedding_intput_dim=num_vocab['brand_id'])
]
features = [
    mf.SparseFeature(name='user_id', lookup_table_path=vocab_paths['user_id'], embedding_intput_dim=num_vocab['user_id']),
    mf.SparseFeature(name='user_type', lookup_table_path=vocab_paths['user_type'], embedding_intput_dim=num_vocab['user_type']),
    mf.SparseFeature(name='member_level', lookup_table_path=vocab_paths['member_level'], embedding_intput_dim=num_vocab['member_level']),
    mf.SparseFeature(name='hist_item_id', input_shape=(10,), embedding_name='item_id', masked=True, attribute_features=item_id_attribute_features),
    mf.DenseFeature(name='hist_len', input_dtype=tf.int32),
    mf.SparseFeature(name='item_id', masked=True,
                     attribute_features=item_id_attribute_features),
    mf.SparseFeature(name='neg_item_id', input_shape=(4,), embedding_name='item_id', masked=True, attribute_features=item_id_attribute_features),


]


def test_flatten_features():
    print(mf.flatten_features(features=features))


def test_create_embedding_layers():
    print(mf.create_embedding_layers(features))


def test_create_lookup_table_layers():
    print(mf.create_lookup_table_layers(features))


def test_get_feature_names():
    print(mf.get_feature_names(features))
    print(mf.get_feature_names(features, mf.SparseFeature))


def test_create_split_string_layers():
    print(mf.create_split_string_layers(features))


class TestFeatureBuilder():
    def setup_class(self) -> None:
        self.fb = mf.FeatureBuilder(features)

    def teardown_class(self) -> None:
        self.fb = None

    def test_get_dense_inputs(self):
        print(self.fb.get_dense_inputs())

    def test_get_sparse_embeddings(self):
        print(self.fb.get_sparse_embeddings())
