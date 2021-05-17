"""main program
"""
from tensorflow import make_tensor_proto
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_serving.apis import predict_pb2
from datetime import datetime
import subprocess
import os
import tensorflow as tf
import pickle
from tensorflow.keras.regularizers import L2
import faiss
import tensorflow as tf
from mind.v1 import model as mind_model
from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_enum('env', 'prod', ['prod', 'dev'], 'run environment')
flags.DEFINE_string('run_mode', 'train', 'run mode: train|gen_serving')
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

with open(meta_path, 'rb') as f:
    vocab_paths = pickle.load(f)
    num_vocabs = pickle.load(f)
    context_length, neg_sample_num = pickle.load(f)

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


def create_model(dynamic_k=False):
    return mind_model.MIND(vocab_paths,
                           num_vocabs,
                           embedding_dims,
                           hist_max_len=context_length,
                           num_sampled=neg_sample_num,
                           dynamic_k=dynamic_k,
                           k_max=5,
                           p=10.0,
                           embeddings_regularizer=None,
                           kernal_regularizer=L2(0.0001),
                           user_dnn_hidden_units=(embedding_dim,))


def gen_prod_environment():
    pass


def gen_dev_environment():
    pass


def main(_):
    if FLAGS.env == 'prod':
        gen_prod_environment()
    else:
        gen_dev_environment()
    if FLAGS.run_mode == 'train':
        logging.debug('training mode')
        model = create_model(False)
        train_model(model)
    elif FLAGS.run_mode == 'gen_serving':
        logging.debug('gen_serving mode')
        model = create_model(True)
        gen_serving_model(model)


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


def train_model(model: mind_model.MIND):

    train_dataset = gen_dataset(train_path, batch_size)
    test_dataset = gen_dataset(test_path, test_batch_size)
    model.compile()
    model.train(train_dataset, test_dataset, epochs=epochs, steps_per_epoch=None, checkpoint_path=checkpoint_path,
                checkpoint_frequency=checkpoint_frequency, restore_latest=False, monitor='val_acurracy', mode='max')


def gen_serving_model(model: mind_model.MIND):
    model.compile(optimizer="adam")
    model.load_weights(checkpoint_path, True)
    serving_model = model.get_serving_model()
    tfserving_model_name = FLAGS.tfserving_model_name
    date = datetime.now().strftime(format="%Y%m%d")
    saved_model_path = 'model/' + tfserving_model_name + "/" + date
    logging.info('saved_model_path: %s .', saved_model_path)
    model.save_model(saved_model_path, model=serving_model)
    if FLAGS.gen_warm_up_requests:
        gen_warm_up_requests(saved_model_path, tfserving_model_name)


def gen_warm_up_requests(export_dir, model, signature_name='serving_default'):

    def make_assets_dir(export_dir):
        assets_dir = os.path.join(export_dir, 'assets.extra')
        if tf.io.gfile.isdir(assets_dir):
            tf.io.gfile.rmtree(assets_dir)
        tf.io.gfile.makedirs(assets_dir)
        return assets_dir

    assets_dir = make_assets_dir(export_dir)
    with tf.io.TFRecordWriter(os.path.join(assets_dir, "tf_serving_warmup_requests")) as writer:
        test_dataset = gen_dataset(test_path, test_batch_size)
        for i, d in enumerate(test_dataset.as_numpy_iterator()):
            print('write index: ', i)
            request = predict_pb2.PredictRequest()
            request.model_spec.name = model
            request.model_spec.signature_name = signature_name
            request.inputs['user_id'].CopyFrom(make_tensor_proto(d['user_id'], shape=[test_batch_size, 1]))
            request.inputs['user_type'].CopyFrom(make_tensor_proto(d['user_type'], shape=[test_batch_size, 1]))
            request.inputs['member_level'].CopyFrom(make_tensor_proto(d['member_level'], shape=[test_batch_size, 1]))
            request.inputs['member_level'].CopyFrom(make_tensor_proto(d['member_level'], shape=[test_batch_size, 1]))
            request.inputs['hist_item_id'].CopyFrom(make_tensor_proto(d['hist_item_id'], shape=d['hist_item_id'].shape))
            request.inputs['hist_len'].CopyFrom(make_tensor_proto(d['hist_len'], shape=[test_batch_size, 1]))
            request.inputs['items_id'].CopyFrom(make_tensor_proto(d['item_id'], shape=[test_batch_size, 1]))

            log = prediction_log_pb2.PredictionLog(
                predict_log=prediction_log_pb2.PredictLog(request=request))
            writer.write(log.SerializeToString())


if __name__ == '__main__':

    app.run(main)
