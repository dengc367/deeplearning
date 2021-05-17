"""main program
"""

import numpy as np
import json
import requests
import grpc
import time
from datetime import datetime
import subprocess
import os
import tensorflow as tf
import pickle
from tensorflow.keras.regularizers import L2
from tensorflow import make_tensor_proto
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2, prediction_log_pb2
from absl import app, flags, logging
import pandas as pd
import faiss
import mind.features as mf
from mind import model as mind_model

FLAGS = flags.FLAGS
flags.DEFINE_enum('env', 'prod', ['prod', 'dev'], 'run environment')
flags.DEFINE_string('run_mode', 'train', 'run mode: train|gen_serving|test_serving')
flags.DEFINE_string('tfserving_model_name', 'mind_model', 'model_name in tfserving config file')
flags.DEFINE_boolean('gen_warm_up_requests', False, 'is gen_warm_up_requests?')

prefix = 'mind'

train_path = "data/train.tfrecord"
test_path = "data/test.tfrecord"


def shellCmd(cmd):
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.readline().decode("utf8").replace("\n", "")


HDFS_URL = shellCmd("hdfs getconf -confKey fs.defaultFS")
HDFS_WORK_DIRECTORY = HDFS_URL + "/user/" + os.environ["USER"]
train_path = HDFS_WORK_DIRECTORY + "/" + prefix + "/" + train_path + "/part-*"
test_path = HDFS_WORK_DIRECTORY + "/" + prefix + "/" + test_path + "/part-*"

meta_path = "meta/data_meta.pkl"
items_info_path = 'meta/items_info.csv'

model_meta_path = "meta/model_meta.pkl"
checkpoint_dir = 'model/saved_ckpt_path'
checkpoint_path = checkpoint_dir + '/cp-{epoch:04d}.ckpt'
checkpoint_frequency = 'epoch'
epochs = 4
buffer_size = 1024
batch_size = 128
test_batch_size = 1024
embedding_dim = 32

with open(meta_path, 'rb') as f:
    vocab_paths = pickle.load(f)
    num_vocabs = pickle.load(f)
    context_length, neg_sample_num = pickle.load(f)
print('vocab_paths: ', vocab_paths)
print('num_vocab: ', num_vocabs)
print('context_length: ', context_length, ', neg_sample_num: ', neg_sample_num)


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


def create_model(dynamic_k=False, k_max=5):

    model = mind_model.MIND(features,
                            hist_max_len=context_length,
                            num_sampled=neg_sample_num,
                            dynamic_k=dynamic_k,
                            k_max=k_max,
                            p=10.0,
                            kernal_regularizer=kernal_regularizer,
                            user_dnn_hidden_units=(embedding_dim*2, embedding_dim,))
    # model.summary()
    return model


def main(_):

    logging.debug('run_mode: %s', FLAGS.run_mode)

    model = create_model()
    model.compile()
    train_dataset = gen_dataset(train_path, batch_size)
    test_dataset = gen_dataset(test_path, test_batch_size)

    if FLAGS.run_mode == 'train':
        train_model(model, train_dataset, test_dataset)
    elif FLAGS.run_mode == 'test':
        test_model()
    elif FLAGS.run_mode == 'gen_serving':
        gen_serving_model(model, test_dataset)
    elif FLAGS.run_mode == 'test_serving':
        test_serving_api(test_dataset)


def gen_dataset(input_path, batch):
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
        example['hist_item_id'] = example.pop('hist_ids')
        example['hist_len'] = tf.cast(example['hist_len'], tf.int32)
        example['item_id'], example['neg_item_id'] = tf.split(example.pop('label_ids'), [1, neg_sample_num], axis=-1)
        return example
    input_file_pattern = tf.data.Dataset.list_files(input_path)
    dataset = tf.data.TFRecordDataset(input_file_pattern).map(decode).prefetch(batch * 10).shuffle(batch*10)
    dataset = dataset.batch(batch)
    return dataset


def test_serving_gprc_api(d, host='10.254.64.251', port='8504', model='mind_user_similarity', signature_name='serving_default'):

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

    result = stub.Predict(request, 10.0)

    end = time.time()
    time_diff = end - start
    logging.info('time elapased: {}'.format(time_diff))

    # Reference:
    # How to access nested values
    # https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
    indices = result.outputs['tf.argsort']
    probs = result.outputs['tf.sort']
    indices = tf.constant(indices.int_val, dtype=tf.int32, shape=tf.TensorShape(indices.tensor_shape))
    probs = tf.constant(probs.float_val, dtype=tf.float32, shape=tf.TensorShape(probs.tensor_shape))
    logging.info(indices, probs)


def test_serving_http_api(d):

    def default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

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
    payload = json.dumps(payload, default=default)
    logging.debug((payload))

    r = requests.post('http://10.254.64.251:8505/v1/models/mind_user_similarity:predict', data=payload)

    pred = json.loads(r.content.decode('utf-8'))
    logging.info(pred['outputs'])


def test_serving_api(test_dataset):
    it = test_dataset.as_numpy_iterator()
    d = next(it)
    test_serving_http_api(d)
    test_serving_gprc_api(d)


def train_model(model: mind_model.MIND, train_dataset, test_dataset):

    model.train(train_dataset, test_dataset, epochs=epochs, steps_per_epoch=None, checkpoint_path=checkpoint_path,
                checkpoint_frequency=checkpoint_frequency, restore_latest=False, monitor='val_acurracy', mode='max')


def test_model(topk=100, k_max=5):
    model = create_model(True, k_max)
    model.compile()
    model.load_weights(checkpoint_path, True)
    user_embedding_model = model.get_user_model()
    item_embedding_model = model.get_item_model()
    test_dataset = gen_dataset(test_path, test_batch_size)

    # df_item_info = pd.read_csv(items_info_path, header=0, index_col=0).reset_index(drop=True)
    df_item_id = pd.read_csv(vocab_paths['item_id'], names=['id', 'item_id'], index_col='id')
    df_product_id = pd.read_csv(vocab_paths['product_id'], names=['id', 'product_id'], index_col='id')
    item_ids = np.array(df_item_id['item_id']).astype('str')
    product_ids = np.array(df_product_id['product_id']).astype(int)
    indices = np.array(df_product_id.index).astype(int)

    product_item_id_dict = dict(zip(product_ids, item_ids))
    item_product_id_dict = dict(zip(item_ids, product_ids))

    product_index_dict = dict(zip(product_ids, indices))
    item_index_dict = dict(zip(item_ids, indices))
    item_size = len(df_item_id) + 1

    item_embs = item_embedding_model.predict({'item_id': item_ids})
    item_embs = np.squeeze(item_embs)

    quantizer = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, int(np.sqrt(len(product_ids))), faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(item_embs)
    index.train(item_embs)
    logging.info('index trained: %s.', index.is_trained)
    index.add_with_ids(item_embs, product_ids)

    def recall_N(y_true, y_pred, N=50):
        return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)

    recall_scores = []
    lines = 0
    recall_func = tf.metrics.Recall()
    recall_func2 = []
    inter_1s = []
    inter_2s = []
    for i, d in enumerate(test_dataset.as_numpy_iterator()):

        user_embs = user_embedding_model.predict(d)
        faiss.normalize_L2(user_embs)

        user_embs = user_embs.reshape((-1, user_embs.shape[-1]))
        _, search_product_ids = index.search(user_embs, topk)
        search_product_ids = search_product_ids.reshape((-1, topk*k_max))
        b_size = search_product_ids.shape[0]
        time1 = time.time()
        for j in range(b_size):
            search_product_id = search_product_ids[j]
            item_id_j = d['item_id'][j]
            search_product_id = search_product_id.reshape([-1]).tolist()
            grouptruth_item_id = item_id_j.astype(str).tolist()[0]
            grouptruth_product_id = item_product_id_dict[grouptruth_item_id]
            score = recall_N([grouptruth_product_id], search_product_id, topk*k_max)
            recall_scores.append(score)
        time2 = time.time()
        # for tf.metrics.Recall, Be careful, performance decayed when use tf.metrics.Recall
        # search_product_ids_vec = np.vectorize(lambda x: product_index_dict.get(x, 0), otypes=[int])(search_product_ids)  # [B, 100*5]
        # item_id_vec = np.vectorize(item_index_dict.get, otypes=[int])(d['item_id'].astype(str))  # [B, 1]
        # search_product_ids_vec = tf.identity(search_product_ids_vec)
        # item_id_vec = tf.identity(item_id_vec)
        # # y_true: one hot
        # y_true_one_hot = tf.reduce_max(tf.one_hot(item_id_vec, depth=item_size, on_value=1, dtype=tf.int32, axis=-1), axis=-2)
        # # y_pred: multi hot
        # y_pred_one_hot = tf.reduce_max(tf.one_hot(search_product_ids_vec, depth=item_size, on_value=1, dtype=tf.int32, axis=-1), axis=-2)
        # recall_func.update_state(y_true_one_hot, y_pred_one_hot)
        # recall_result = (tf.reduce_sum(y_true_one_hot * y_pred_one_hot)/y_true_one_hot.shape[0]).numpy()
        # recall_func2.append(recall_result)
        time3 = time.time()
        inter_1, inter_2 = time2-time1, time3-time2
        inter_1s.append(inter_1)
        inter_2s.append(inter_2)
        # end

        lines += b_size
        if i % 10 == 0:
            logging.info('inter_1, %f, inter_2: %f', np.mean(inter_1s), np.mean(inter_2s))
            logging.info('batch num: %d,  size: %d, batch recall ratio: %f, tf.metrics.Recall Ratio: %f, tf batch recall ratio: %f',
                         i, lines, np.mean(recall_scores), recall_func.result().numpy(), np.mean(recall_func2))
    logging.info('all size: %d, recall ratio: %f, tf.metrics.Recall Ratio: %f, tf batch recall ratio: %f', lines, np.mean(recall_scores), recall_func.result().numpy(), np.mean(recall_func2))


def gen_serving_model(model: mind_model.MIND, dataset):
    model.load_weights(checkpoint_path, True)
    serving_model = model.get_serving_model()
    tfserving_model_name = FLAGS.tfserving_model_name
    date = datetime.now().strftime(format="%Y%m%d")
    saved_model_path = 'model/' + tfserving_model_name + "/" + date
    logging.info('saved_model_path: %s .', saved_model_path)
    model.save_model(saved_model_path, model=serving_model)
    if FLAGS.gen_warm_up_requests:
        it = dataset.as_numpy_iterator()
        d = next(it)
        gen_warm_up_requests(saved_model_path, tfserving_model_name, d)


def gen_warm_up_requests(export_dir, model, d, signature_name='serving_default'):

    def make_assets_dir(export_dir):
        assets_dir = os.path.join(export_dir, 'assets.extra')
        if tf.io.gfile.isdir(assets_dir):
            tf.io.gfile.rmtree(assets_dir)
        tf.io.gfile.makedirs(assets_dir)
        return assets_dir

    assets_dir = make_assets_dir(export_dir)
    with tf.io.TFRecordWriter(os.path.join(assets_dir, "tf_serving_warmup_requests")) as writer:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model
        request.model_spec.signature_name = signature_name
        request.inputs['user_id'].CopyFrom(make_tensor_proto(d['user_id']))
        request.inputs['user_type'].CopyFrom(make_tensor_proto(d['user_type']))
        request.inputs['member_level'].CopyFrom(make_tensor_proto(d['member_level']))
        request.inputs['hist_item_id'].CopyFrom(make_tensor_proto(d['hist_item_id']))
        request.inputs['hist_len'].CopyFrom(make_tensor_proto(d['hist_len']))
        request.inputs['item_id'].CopyFrom(make_tensor_proto(np.concatenate([d['item_id'], d['neg_item_id']], axis=-1)))
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())


if __name__ == '__main__':
    app.run(main)
