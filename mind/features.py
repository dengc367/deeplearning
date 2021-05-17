import json
from mind.layers import MaskZero, ListMeanPooling
from tensorflow.keras.layers import concatenate
from mind.layers import LookupTable, SplitString
from typing import List, Mapping, Sequence, Union
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding


class Feature:
    def __init__(self, name, input_shape, input_dtype, masked=False):
        self.name = name
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.masked = masked

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, o: object) -> bool:
        return self.name == o.name

    def __str__(self) -> str:
        return json.dumps(self.__dict__, default=lambda x: str(x))


class SparseFeature(Feature):
    def __init__(self, name, vocab_size, embedding_dim, input_shape=(1,), input_dtype=tf.string,
                 masked=False,
                 lookup_table_path=None,
                 attribute_features=[],
                 attribute_features_separator='_',
                 attribute_features_combiner='mean',
                 embedding_name=None,
                 embeddings_initializer='glorot_normal',
                 embeddings_regularizer=None
                 ):
        if isinstance(attribute_features, list) and len(attribute_features) > 0 and input_dtype != tf.string:
            raise "when attribute_features is not empty, input_dtype must be tf.string"
        super().__init__(name, input_shape, input_dtype, masked)
        self.lookup_table_path = lookup_table_path
        self.attribute_features = attribute_features
        self.attribute_features_separator = attribute_features_separator
        self.attribute_features_combiner = attribute_features_combiner

        self.embedding_name = name if embedding_name is None else embedding_name
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer


class DenseFeature(Feature):
    def __init__(self, name, input_shape=(1,), input_dtype=tf.int32):
        assert input_dtype != tf.string
        super().__init__(name, input_shape, input_dtype)


def create_inputs(features, to_list=False) -> Union[Mapping[str, Input], Sequence[Input]]:
    input_layers = {}
    for feature in features:
        input_layers[feature.name] = Input(name=feature.name, shape=feature.input_shape, dtype=feature.input_dtype)
    if to_list:
        return list(input_layers.values())
    return input_layers


def flatten_features(features: List[Feature], to_list=False):
    feature_map = {}
    for f in features:
        if f.name not in feature_map:
            feature_map[f.name] = f
        if isinstance(f, SparseFeature) and isinstance(f.attribute_features, list):
            for att_f in f.attribute_features:
                if att_f.name not in feature_map:
                    feature_map[att_f.name] = att_f
    if to_list:
        return list(feature_map.values())
    return feature_map


def create_embedding_layers(features: List[Feature], to_list=False) -> Union[Mapping[str, Embedding], Sequence[Embedding]]:
    embedding_layers = {}
    for feature in flatten_features(features, True):
        if isinstance(feature, SparseFeature) and feature.embedding_name not in embedding_layers and not feature.masked:
            embedding_layers[feature.embedding_name] = Embedding(feature.vocab_size, feature.embedding_dim, embeddings_initializer=feature.embeddings_initializer,
                                                                 embeddings_regularizer=feature.embeddings_regularizer, name='embedding_' + feature.embedding_name, mask_zero=True)
    if to_list:
        return list(embedding_layers.values())
    return embedding_layers


def create_lookup_table_layers(features: List[Feature], to_list=False):
    lookup_table_layers = {}
    for feature in flatten_features(features, True):
        if isinstance(feature, SparseFeature) and feature.lookup_table_path is not None and not feature.masked:
            lookup_table_layers[feature.embedding_name] = LookupTable(feature.lookup_table_path, name='lookup_'+feature.embedding_name)
    if to_list:
        return list(lookup_table_layers.values())
    return lookup_table_layers


def get_feature_names(features, return_feature_type=None):
    feature_names = []
    for feature in flatten_features(features, True):
        if return_feature_type is None or isinstance(feature, return_feature_type):
            feature_names.append(feature.name)
    return feature_names


def create_split_string_layers(features: List[Feature], to_list=False):
    split_string_layers = {}
    for feature in flatten_features(features, True):
        if isinstance(feature, SparseFeature) and isinstance(feature.attribute_features, list) and len(feature.attribute_features) > 0 and feature.embedding_name not in split_string_layers:
            attribute_features_names = get_feature_names(feature.attribute_features)
            split_string_layers[feature.embedding_name] = SplitString(attribute_features_names, sep=feature.attribute_features_separator, name='split_'+feature.embedding_name)
    if to_list:
        return list(split_string_layers.values())
    return split_string_layers


class FeatureBuilder:
    def __init__(self, features: List[Feature]) -> None:
        self.features = features
        self.feature_map = flatten_features(features)

        self.input_layers = create_inputs(features)

        self.split_string_layers = create_split_string_layers(features)
        self.lookup_table_layers = create_lookup_table_layers(features)
        self.embedding_layers = create_embedding_layers(features)

    def get_inputs(self, to_list=False):
        input_layers = self.input_layers
        if to_list:
            return list(input_layers.values())
        return input_layers

    def lookup_embeddings(self, return_features: List[Feature] = [], to_list=False):
        dense_inputs = self.get_dense_inputs()
        sparse_embeddings = self.get_sparse_embeddings()
        return_embeddings = dict(list(dense_inputs.items()) + list(sparse_embeddings.items()))
        if isinstance(return_features, list) and len(return_features) > 0:
            return_embeddings = {f.name: return_embeddings[f.name] for f in return_features}
        if to_list:
            return list(return_embeddings.values())
        return return_embeddings

    def get_dense_inputs(self, to_list=False):
        dense_inputs = {}
        for feature in self.features:
            if isinstance(feature, DenseFeature):
                dense_inputs[feature.name] = self.input_layers[feature.name]
        if to_list:
            return list(dense_inputs.values())
        return dense_inputs

    def get_sparse_embeddings(self, to_list=False):
        embeddings = {}
        for feature in self.features:
            if not isinstance(feature, SparseFeature):
                continue
            if isinstance(feature.attribute_features, list) and len(feature.attribute_features) > 0:
                embedding = self.get_item_embedding(self.input_layers, feature)
            else:
                embedding = self.lookup_embedding(self.input_layers, feature)
            if embedding is not None:
                embeddings[feature.name] = embedding
        if to_list:
            return list(embeddings.values())
        return embeddings

    def lookup_embedding(self, inputs, feature: SparseFeature, mask_zero=False):
        name = feature.name
        embedding_name = feature.embedding_name
        if feature.masked:
            return None
        ids = self.lookup_table_layers[embedding_name](inputs[name])
        embs = self.embedding_layers[embedding_name](ids)
        if mask_zero:
            return MaskZero(name='masked_emb_'+name)(embs)
        return embs

    def get_item_embedding(self, inputs, feature: SparseFeature):
        name = feature.name
        embedding_name = feature.embedding_name
        combiner = feature.attribute_features_combiner
        embeds = {}
        if not feature.masked:
            embeds[name] = self.lookup_embedding(inputs, feature)

        attr_inputs = self.split_string_layers[embedding_name](inputs[name])
        for attr_name in attr_inputs.keys():
            attr_feature = self.feature_map[attr_name]
            if not attr_feature.masked:
                embeds[attr_name] = self.lookup_embedding(attr_inputs, attr_feature)

        if combiner == 'mean':
            embed = ListMeanPooling(name='mean_'+name)(embeds.values())
        elif combiner == 'concat':
            embed = concatenate(embeds.values(), name='concat_' + name)
        return embed
