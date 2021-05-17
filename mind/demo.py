#  %%

from re import X
from tensorflow.keras.regularizers import L2
from scipy import spatial
import faiss
import tensorflow as tf
from mind import model as mind_model
import sys
import importlib
import pandas as pd
import numpy as np
import pickle


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
    epochs = 2
    buffer_size = 1024
    batch_size = 64
    test_batch_size = 1
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
if sys.modules['mind.model'] is not None:
    importlib.reload(sys.modules['mind.model'])


model = mind_model.MIND(vocab_paths,
                        num_vocabs,
                        embedding_dims,
                        hist_max_len=context_length,
                        num_sampled=neg_sample_num,
                        dynamic_k=False,
                        k_max=5,
                        p=10.0,
                        embeddings_regularizer=None,
                        kernal_regularizer=L2(0.0001),
                        user_dnn_hidden_units=(embedding_dim,))
model.summary()
# %%
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
model.compile()
model.train(train_dataset, test_dataset, epochs=epochs, steps_per_epoch=None, checkpoint_path=checkpoint_path,
            checkpoint_frequency=checkpoint_frequency, restore_latest=False, monitor='val_acurracy', mode='max')

# %%
model.compile()
model.train(train_dataset, test_dataset, epochs=3, steps_per_epoch=None, checkpoint_path=checkpoint_path,
            checkpoint_frequency=checkpoint_frequency, restore_latest=True, monitor='val_acurracy', mode='max')
# %%
model.compile(optimizer="adam")
model.load_weights(checkpoint_path, True)
# model.train(train_dataset, test_dataset, epochs, checkpoint_path, checkpoint_frequency)
# %%
user_embedding_model = model.get_user_model()
item_embedding_model = model.get_item_model()
# user_embedding_model.summary()
# item_embedding_model.summary()

serving_model = model.get_serving_model()
# serving_model.summary()


# %%

# %%
# data_item_ids = set()
# for i, d in enumerate(train_dataset.as_numpy_iterator()):
#     a = np.reshape(np.concatenate([d['hist_item_id'], d.pop('item_id'), d.pop('neg_item_id')], axis=-1), -1)
#     # data_item_ids.add(list(a))
#     # print(a)
#     a = set(map(lambda x: x.decode('utf-8'), a))
#     # data_item_ids.add(a)
#     data_item_ids.update(a)
#     # print(a)
#     # if i > 3:
#     #     break
# print(len(data_item_ids))
# print(list(data_item_ids)[:10])
# %%
df_item_info = pd.read_csv('dev/meta/items_info.csv', header=0, index_col=0).reset_index(drop=True)
# df_item_info.head()

df_item_id = pd.read_csv(vocab_paths['item_id'], names=['id', 'item_id'], index_col='id')
item_ids = np.array(df_item_id['item_id'], dtype='str')
product_ids = np.array(list(map(lambda x: x.split('_')[0], item_ids)),
                       dtype=np.int64)
# print(len(df_item_id))
item_ids_dataset = {'item_id': item_ids}
# %%
for i, d in enumerate(test_dataset.as_numpy_iterator()):
    # print(d, d['user_id'].shape)
    pred_item_ids = item_ids
    # pred_item_ids = np.array(['25947_101_400102199_400102779_703', '100499_400102107_400102163_400102557_591'])
    d['item_ids'] = np.repeat(np.expand_dims(pred_item_ids, 0), d['user_id'].shape[0], axis=0)
    items_id = list(map(lambda x: x.decode('utf-8'), np.squeeze(np.concatenate([d['hist_item_id'], d.pop('item_id')], axis=-1))))
    df_xx_items_id = df_item_info.set_index('item_id')
    print(items_id)
    print(df_xx_items_id.loc[items_id].reset_index(drop=True)[['product_id', 'product_name', 'category_name1', 'category_name2', 'category_name3', 'brand_name']])
    probs, indices = serving_model.predict(d)
    # print(probs[:, :, :10])
    print(indices[:, :, :10], probs.shape)
    # pids = np.squeeze(indices[:, 1, :10])
    # print(pids)
    for i in range(indices.shape[1]):
        print(df_item_info.loc[np.squeeze(indices[:, i, :10])][['product_id', 'product_name', 'category_name1', 'category_name2', 'category_name3', 'brand_name']])

    # if i >= 10:
    break
# %%


def recall_N(y_true, y_pred, N=50):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)


# %%
item_embs = item_embedding_model.predict(item_ids_dataset)
item_embs = np.squeeze(item_embs)
item_embs.shape

quantizer = faiss.IndexFlatIP(embedding_dim)
index = faiss.IndexIVFFlat(quantizer, embedding_dim, int(np.sqrt(len(product_ids))), faiss.METRIC_INNER_PRODUCT)
faiss.normalize_L2(item_embs)
index.train(item_embs)
print(index.is_trained)
# index.add(item_embs)
index.add_with_ids(item_embs, product_ids)

# index.add_with_ids(item_embs, product_ids)
# index = faiss.index_factory(embedding_dim, "Flat", faiss.METRIC_INNER_PRODUCT)
# faiss.normalize_L2(item_embs)
# index.add(item_embs)

# index = faiss.IndexFlatIP(embedding_dim)
# faiss.normalize_L2(item_embs)
# index.add(item_embs)

print(index.ntotal)


# %%
faiss.write_index(index, 'dev/data/faiss_index.bin')
index = faiss.read_index('dev/data/faiss_index.bin')
# %%
print(product_ids[:3], product_ids[5255], product_ids[4681], product_ids[5104])
print(item_ids[:2])

ddd, iii = index.search(item_embs[:2], 10)
ddd, iii
# %%
s = []
lines = 0
hits = 0
topk = 100
for d in (test_dataset.as_numpy_iterator()):
    # print(d['user_id'].shape)
    # print(d['item_id'].shape)
    batch_size = d['user_id'].shape[0]
    pred_user_embs = user_embedding_model.predict(d)

    faiss.normalize_L2(pred_user_embs)
    if len(pred_user_embs.shape) == 3:
        nq, nk = pred_user_embs.shape[0], pred_user_embs.shape[1]
        result_heap = faiss.ResultHeap(nq=nq, k=topk)
        ids_list = []
        for i in range(nk):
            ni = np.ascontiguousarray(pred_user_embs[:, i, :])
            dists, ids = index.search(ni, topk)
            result_heap.add_result(D=dists, I=ids)
            ids_list.append(ids)
        result_heap.finalize()
        dists, ids = result_heap.D, result_heap.I
        ids2 = np.concatenate(ids_list, axis=0)
    elif len(pred_user_embs.shape) == 2:
        dists, ids = index.search(pred_user_embs, topk)
    else:
        raise('shape rank is not 2 or 3')

    uids = d['user_id']
    iids = d['item_id']
    groudtruth_pids = list(map(lambda x: int((x[0].decode("utf-8").split('_')[0])), iids))
    for i in range(batch_size):
        groudtruth_pid = groudtruth_pids[i]
        pred_pids = list(ids[i])
        pred_pids2 = list(ids2[i])
        # print(item_id, pred_item_ids)
        # print(type(item_id), type(pred_item_ids))
        recall_score = recall_N([groudtruth_pid], pred_pids, topk * 5)
        s.append(recall_score)
        lines += 1
        if groudtruth_pid in pred_pids2:
            hits += 1
    # print(I2.shape)
    if lines > 1000:
        break
print("recall: ", np.mean(s))
print("hit ratio: ,", hits / lines, ", hits: ", hits, "lines: ", lines)

# %%

# %%

# %%


# %%
dataSetI = [.1, .2, .3]
dataSetII = [.4, .5, .6]
# dataSetII = [.1, .2, .3]

x = np.array([dataSetI]).astype(np.float32)
q = np.array([dataSetII]).astype(np.float32)
index = faiss.index_factory(3, "Flat", faiss.METRIC_INNER_PRODUCT)
index.ntotal
faiss.normalize_L2(x)
index.add(x)
faiss.normalize_L2(q)
distance, index = index.search(q, 5)
print('Distance by FAISS:{}'.format(distance))

# To Tally the results check the cosine similarity of the following example


result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
print('Distance by FAISS:{}'.format(result))
# %%
