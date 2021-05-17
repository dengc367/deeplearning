#  %%
import grpc
import time
import json
import requests
import mind.features as mf
import sys
import importlib
import numpy as np
import pickle
from tensorflow.keras.regularizers import L2
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow import make_tensor_proto
from mind import model as mind_model


# %%
mode = 'dev'
if mode == 'dev':
    prefix = mode
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

else:
    prefix = 'mind'

    train_path = "data/train.tfrecord"
    test_path = "data/test.tfrecord"

    import os
    import subprocess

    def shellCmd(cmd):
        return subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE).stdout.readline().decode("utf8").replace(
                "\n", "")

    HDFS_URL = shellCmd("hdfs getconf -confKey fs.defaultFS")
    HDFS_WORK_DIRECTORY = HDFS_URL + "/user/" + os.environ["USER"]
    train_path = HDFS_WORK_DIRECTORY + "/" + prefix + "/" + train_path + "/part-*"
    test_path = HDFS_WORK_DIRECTORY + "/" + prefix + "/" + test_path + "/part-*"

    meta_path = "meta/data_meta.pkl"
    model_meta_path = "meta/model_meta.pkl"
    checkpoint_dir = 'model/saved_ckpt_path'
    checkpoint_path = checkpoint_dir + '/cp-{epoch:04d}.ckpt'
    checkpoint_frequency = 'epoch'
    epochs = 4
    buffer_size = 1024
    batch_size = 64
    test_batch_size = 128
    embedding_dim = 32

    with open(meta_path, 'rb') as f:
        vocab_paths = pickle.load(f)
        num_vocabs = pickle.load(f)
        context_length, neg_sample_num = pickle.load(f)

print('vocab_paths: ', vocab_paths)
print('num_vocab: ', num_vocabs)
print('context_length: ', context_length, ', neg_sample_num: ', neg_sample_num)
num_vocabs = {k: v + 1 for k, v in num_vocabs.items()}
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
if sys.modules.get('mind.model_v2') is not None:
    importlib.reload(sys.modules['mind.model_v2'])

embeddings_regularizer = L2(0.0001)
kernal_regularizer = L2(0.0001)


def gen_sparse_feature(name, embedding_name=None, **kwargs):
    embedding_name = name if embedding_name is None else embedding_name
    return mf.SparseFeature(name=name, vocab_size=num_vocabs[embedding_name], embedding_dim=embedding_dims[embedding_name],
                            lookup_table_path=vocab_paths[embedding_name], embeddings_regularizer=embeddings_regularizer, **kwargs)


item_id_attribute_features = [
    gen_sparse_feature('product_id'),
    gen_sparse_feature('first_class_id'),
    gen_sparse_feature('second_class_id'),
    gen_sparse_feature('third_class_id'),
    gen_sparse_feature('brand_id'),
]

user_features = [
    gen_sparse_feature('user_id'),
    gen_sparse_feature('user_type'),
    gen_sparse_feature('member_level'),
    gen_sparse_feature('hist_item_id', embedding_name='item_id', input_shape=(context_length,), masked=True, attribute_features=item_id_attribute_features),
    mf.DenseFeature(name='hist_len', input_dtype=tf.int32),
]
item_features = [
    gen_sparse_feature('item_id', masked=True, attribute_features=item_id_attribute_features),
]
neg_item_features = [
    gen_sparse_feature('neg_item_id', embedding_name='item_id', input_shape=(neg_sample_num,), masked=True, attribute_features=item_id_attribute_features),
]

features = (user_features, item_features, neg_item_features)

model = mind_model.MIND(features,
                        hist_max_len=context_length,
                        num_sampled=neg_sample_num,
                        dynamic_k=False,
                        k_max=5,
                        p=10.0,
                        kernal_regularizer=kernal_regularizer,
                        user_dnn_hidden_units=(embedding_dim*2, embedding_dim,))
model.summary()

# %%


def gen_dataset(input_path, batch_size):
    def decode(serialized):
        example = tf.io.parse_single_example(
            serialized,
            features={
                'user_id': tf.io.FixedLenFeature([1], tf.string),
                'user_type': tf.io.FixedLenFeature([1], tf.string),
                'member_level': tf.io.FixedLenFeature([1], tf.string),
                'hist_ids': tf.io.FixedLenFeature([context_length], tf.string),
                'hist_len': tf.io.FixedLenFeature([1], tf.int64),

                'label_ids': tf.io.FixedLenFeature([1 + neg_sample_num], tf.string),
            })
        # http://10.254.8.108:8888/lab/tree/mind/gen_dataset.ipynb 这个上面的 hist_ids默认值在前面
        example['hist_item_id'] = tf.reverse(example.pop('hist_ids'), axis=[-1])
        example['hist_len'] = tf.cast(example['hist_len'], tf.int32)
        example['item_id'], example['neg_item_id'] = tf.split(example.pop('label_ids'), [1, 4], axis=-1)
        return example
    input_file_pattern = tf.data.Dataset.list_files(input_path)
    dataset = tf.data.TFRecordDataset(input_file_pattern).map(decode).prefetch(batch_size * 10).shuffle(batch_size*10)
    dataset = dataset.batch(batch_size)
    return dataset


train_dataset = gen_dataset(train_path, batch_size)
test_dataset = gen_dataset(test_path, test_batch_size)

# %%
# model.compile()
# model.train(train_dataset, test_dataset, epochs=1, steps_per_epoch=None, checkpoint_path=checkpoint_path,
#             checkpoint_frequency=checkpoint_frequency, restore_latest=False, monitor='val_acurracy', mode='max')
# %%

model.compile(optimizer="adam")
model.load_weights(checkpoint_path, True)
# model.train(train_dataset, test_dataset, epochs, checkpoint_path, checkpoint_frequency)
# %%
user_embedding_model = model.get_user_model()
item_embedding_model = model.get_item_model()
user_embedding_model.summary()
item_embedding_model.summary()

serving_model = model.get_serving_model()
serving_model.summary()

# %%
it = test_dataset.as_numpy_iterator()
d = next(it)
# %%
# print(d['user_id'].tolist())
# d['hist_len'].tolist()
# d['user_type']
# np.char.decode(d['user_type'].astype(np.bytes_), 'UTF-8').tolist()
np.concatenate([d['item_id'], d['neg_item_id']], axis=-1)
# %%)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


payload = {
    "inputs": {
        "user_id": d['user_id'].astype('U13').tolist(),  # [B] eg: [22][[],[]]
        'user_type': np.char.decode(d['user_type'].astype(np.bytes_), 'UTF-8').tolist(),
        'member_level': np.char.decode(d['member_level'].astype(np.bytes_), 'UTF-8').tolist(),
        'hist_item_id': d['hist_item_id'].astype('U13').tolist(),
        'hist_len': d['hist_len'].tolist(),
        'item_id': np.concatenate([d['item_id'], d['neg_item_id']], axis=-1).astype('U13').tolist(),
    }
}
payload = json.dumps(payload, cls=MyEncoder)
print((payload))

r = requests.post('http://10.254.64.251:8505/v1/models/mind_user_similarity:predict', data=payload)
# r = requests.post('http://10.254.64.89:8503/v1/models/match_ItemsSimilarity:predict', data=payload)
# r = requests.post('http://tf2-serving.prod.chunbo.com/v1/models/match_ItemsSimilarity:predict', data=payload)

pred = json.loads(r.content.decode('utf-8'))
print(pred['outputs'])
# %%


def run(host='10.254.64.251', port='8504', model='mind_user_similarity', signature_name='serving_default'):

    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    start = time.time()

    # Call classification model to make prediction
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    request.inputs['user_id'].CopyFrom(make_tensor_proto(d['user_id']))
    request.inputs['user_type'].CopyFrom(make_tensor_proto(d['user_type']))
    request.inputs['member_level'].CopyFrom(make_tensor_proto(d['member_level']))
    request.inputs['hist_item_id'].CopyFrom(make_tensor_proto(d['hist_item_id']))
    request.inputs['hist_len'].CopyFrom(make_tensor_proto(d['hist_len']))
    request.inputs['item_id'].CopyFrom(make_tensor_proto(np.concatenate([d['item_id'], d['neg_item_id']], axis=-1)))

#     print(request)
    result = stub.Predict(request, 10.0)

    end = time.time()
    time_diff = end - start

    # Reference:
    # How to access nested values
    # https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
    print(result.outputs['tf.argsort'])
    print('time elapased: {}'.format(time_diff))


run()
# %%
