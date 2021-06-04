# %%
from word2vec.model import Word2Vec
import pickle
import subprocess
import os
# from mind.layers import SplitString
# from core.features import SparseFeature


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


meta_path = prefix + "/meta/data_meta.pkl"
model_meta_path = prefix + "/meta/model_meta.pkl"
checkpoint_dir = prefix + '/model/saved_ckpt_path'
checkpoint_path = checkpoint_dir + '/cp-{epoch:04d}.ckpt'
# checkpoint_path = checkpoint_dir + '/cp-0.ckpt'
checkpoint_frequency = 2000
# checkpoint_frequency='epoch'

buffer_size = 1024
batch_size = 128
num_epochs = 4
learning_rate = 0.001


embedding_dims = {
    'item_id': 100,
    'product_id': 100,
    'store_id': 10,
    'brand_id': 20,
    'first_class_id': 10,
    'second_class_id': 10,
    'third_class_id': 20
}


with open(meta_path, 'rb') as f:
    vocab_paths = pickle.load(f)
    vocab_paths = {k: prefix+'/'+v for k, v in vocab_paths.items()}
    print('vocab_paths: ', vocab_paths)
    num_vocab = pickle.load(f)
    print('num_vocab: ', num_vocab)
    context_length, neg_sample_num = pickle.load(f)
    print('context_length: ', context_length, ', neg_sample_num: ', neg_sample_num)
#     session_size, train_size, test_size = pickle.load(f)
#     print('session_size: ', session_size, 'train_size: ', train_size, 'test_size: ', test_size)

# with open(model_meta_path, 'wb') as f:
#     pickle.dump(embedding_dims, f, protocol=pickle.HIGHEST_PROTOCOL)
#     pickle.dump((buffer_size, batch_size, neg_sample_num, num_epochs, learning_rate), f, protocol=pickle.HIGHEST_PROTOCOL)

# embeddings_regularizer = None


# def gen_sparse_feature(name, embedding_name=None, **kwargs):
#     embedding_name = name if embedding_name is None else embedding_name
#     return SparseFeature(name=name, vocab_size=num_vocab[embedding_name], embedding_dim=embedding_dims[embedding_name],
#                          lookup_table_path=vocab_paths[embedding_name], embeddings_regularizer=embeddings_regularizer, **kwargs)


# def get_attr_features(feature_prefix_name):
#     item_id_attribute_features = [
#         gen_sparse_feature(feature_prefix_name+'product_id'),
#         gen_sparse_feature(feature_prefix_name+'site_id'),
#         gen_sparse_feature(feature_prefix_name+'first_class_id'),
#         gen_sparse_feature(feature_prefix_name+'second_class_id'),
#         gen_sparse_feature(feature_prefix_name+'third_class_id'),
#         gen_sparse_feature(feature_prefix_name + 'brand_id'),
#     ]
#     return item_id_attribute_features


# target_feature_name = 'target'
# context_feature_name = 'context'
# target_features = [
#     gen_sparse_feature(target_feature_name, embedding_name=target_feature_name+'item_id', attribute_features=get_attr_features(target_feature_name), attribute_features_combiner='concat'),
# ]
# context_features = [
#     gen_sparse_feature(context_feature_name, embedding_name=context_feature_name + 'item_id', attribute_features=get_attr_features(context_feature_name), attribute_features_combiner='concat'),
# ]
# neg_context_features = [
#     gen_sparse_feature('neg_context', embedding_name=context_feature_name + 'item_id', input_shape=(neg_sample_num,),
#                        attribute_features=get_attr_features(context_feature_name), attribute_features_combiner='concat'),
# ]

# features = (target_features, context_features, neg_context_features)
# model = Word2Vec(features)


# %%

model = Word2Vec(vocab_paths=vocab_paths, num_vocab=num_vocab, embedding_dims=embedding_dims)
model.summary()
# %%

serving_model = model.get_serving_model()
serving_model.summary()

# %%
item_model = model.get_item_model()
item_model.summary()

# %%


# %%
