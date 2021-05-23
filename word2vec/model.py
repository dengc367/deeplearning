import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2
from tensorflow.python.ops.nn_impl import l2_normalize
from core.base_model import BaseModel
from core.layers import SplitString, LookupTable, MaskZero
from word2vec.layers import NCE, StructuralConnenctions


class Word2Vec(BaseModel):
    """ref: Learning and transferring IDs representation in e-commerce
    """

    def __init__(self, vocab_paths, num_vocab, embedding_dims,
                 alpha=1, beta=0.01, num_true=1, num_sampled=2):
        item_feature_name = 'item_id'
        attr_feature_names = ['product_id', 'store_id', 'first_class_id', 'second_class_id', 'third_class_id', 'brand_id']

        self.alpha = alpha
        self.beta = beta
        self.num_true = num_true
        self.num_sampled = num_sampled

        l2_reg = L2(beta) if beta is not None else None

        self.feature_names = [item_feature_name] + attr_feature_names

        self.split_string_layer = SplitString(attr_feats=attr_feature_names, feature_name=item_feature_name, name='split_' + item_feature_name)
        self.vocab_lookup_layer = {name: LookupTable(vocab_paths['item_id'], name='lookup_'+name) for name in self.feature_names}

        def get_embedding(name, prefix, embedings_regularizer=None):
            return Embedding(num_vocab[name], embedding_dims[name], embeddings_initializer='glorot_normal',
                             embeddings_regularizer=embedings_regularizer, name=f'emb_{prefix}_{name}', mask_zero=True)

        self.embedding_params = {name: get_embedding(name, 'target') for name in self.feature_names}
        self.weight_params = {name: get_embedding(name, 'context', embedings_regularizer=l2_reg) for name in self.feature_names}

        self.nce_layer = NCE(self.num_true, self.num_sampled)

        self.structural_connections_layer = StructuralConnenctions(item_feature_name, attr_feature_names, embedding_dims=embedding_dims, l2_reg=l2_reg)

        self.mask_zero_layer = MaskZero()
        model = self._create_model()
        super().__init__(model)

    def get_embeddings(self, x):
        splits = self.split_string_layer(x)
        ids = {name: self.vocab_lookup_layer[name](splits[name]) for name in self.feature_names}
        embeddings = {name: self.embedding_params[name](ids[name]) for name in self.feature_names}  # [N, 1, m1+...+m7]
        embeddings = {name: self.mask_zero_layer(embedding) for name, embedding in embeddings.items()}  # [N, 1, m1+...+m7]
        return embeddings

    def get_weights(self, y):
        splits = self.split_string_layer(y)
        ids = {name: self.vocab_lookup_layer[name](splits[name]) for name in self.feature_names}
        weights = {name: self.weight_params[name](ids[name]) for name in self.feature_names}  # [N, 1, m1+...+m7]
        weights = {name: self.mask_zero_layer(weight) for name, weight in weights.items()}  # [N, 1, m1+...+m7]
        return weights

    def _create_model(self):
        """ create the word2vec model
        """
        inputs = {
            'target': Input(shape=(1,), name='target', dtype=tf.string),
            'context': Input(shape=(self.num_true+self.num_sampled,), name='context', dtype=tf.string),
        }

        target_embeddings = self.get_embeddings(inputs['target'])
        context_weights = self.get_weights(inputs['context'])

        nce_loss = self.nce_layer((target_embeddings, context_weights))
        structural_loss = self.structural_connections_layer(target_embeddings)
        loss = nce_loss + self.alpha * structural_loss

        model = Model(inputs=list(inputs.values()), outputs=loss, name='word2vec_model')
        model.add_loss(loss)

        return model

    def get_serving_model(self):
        inputs = {
            'item_id': Input(shape=(None,), name='item_id', dtype=tf.string),  # [1,m]
            'recommend_item_id': Input(shape=(None,), name='recommend_item_id', dtype=tf.string),  # [1,n]
        }
        item_id_embeddings = self.embedding_and_normalize(inputs['item_id'])

        recommend_item_id_embeddings = self.embedding_and_normalize(inputs['recommend_item_id'])

        probs = tf.einsum('ijk,ilk->il', item_id_embeddings, recommend_item_id_embeddings)
        probs = Lambda(lambda x: x, name='probs')(probs)
        serving_model = Model(inputs=list(inputs.values()), outputs=probs, name='serving_model')
        return serving_model

    def embedding_and_normalize(self, inputs):
        embeddings = self.get_embeddings(inputs)
        embeddings = self.structural_connections_layer(embeddings, training=False)
        embeddings = {name: l2_normalize(embedding, axis=-1, name='normal_' + name) for name, embedding in embeddings.items()}  # [N, 1, m1+...+m7]
        embeddings = tf.concat(list(embeddings.values()), axis=-1)  # [N, m, m1+...+m7]
        embeddings = l2_normalize(embeddings, axis=-1)  # [N, m, M]
        return embeddings

    def get_item_model(self):
        inputs = {
            'item_id': Input(shape=(1,), name='item_id', dtype=tf.string),  # [1,m]
        }
        item_embeddings = self.embedding_and_normalize(inputs['item_id'])
        item_embeddings = Lambda(lambda x: x, name='vectors')(item_embeddings)
        serving_model = Model(inputs=list(inputs.values()), outputs=item_embeddings, name='item_embedding_model')
        return serving_model
