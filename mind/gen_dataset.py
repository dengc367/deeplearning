#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# MIND


# In[1]:


from typing import Mapping, Union, Sequence, Any, Tuple
import random
import subprocess
import os
import pyspark
from pyspark.sql import HiveContext, SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, LongType, MapType, IntegerType, ArrayType, StringType
from pyspark.sql.udf import UserDefinedFunction
import numpy as np
from datetime import datetime, timedelta
import pickle


# In[2]:


#hc = HiveContext(sc)
#date = datetime.now().strftime(format="%F")
spark = SparkSession.builder.appName("gen_dataset").enableHiveSupport().getOrCreate()
sc = spark.sparkContext
hc = HiveContext(sc)
date = datetime.now().strftime(format="%F")
#hc.setConf("spark.sql.shuffle.partitions", "200")
#hc.setConf("spark.default.parallelism", "200")
print(date)


# In[3]:


prefix = 'mind'

train_path = "data/train.tfrecord"
test_path = "data/test.tfrecord"

meta_path = "meta/data_meta.pkl"

items_info_path = 'meta/items_info.csv'
vocab_paths = {
    'item_id': 'meta/item_ids.csv',
    'product_id': 'meta/product_ids.csv',
    'first_class_id': 'meta/first_class_ids.csv',
    'second_class_id': 'meta/second_class_ids.csv',
    'third_class_id': 'meta/third_class_ids.csv',
    'brand_id': 'meta/brand_ids.csv',

    'user_id': 'meta/user_ids.csv',
    'user_type': 'meta/user_types.csv',
    'member_level': 'meta/member_levels.csv',

}


def shellCmd(cmd):
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE
                            ).stdout.readline().decode("utf8").replace("\n", "")


HDFS_URL = shellCmd("hdfs getconf -confKey fs.defaultFS")
HDFS_WORK_DIRECTORY = HDFS_URL+"/user/"+os.environ["USER"]
train_path = HDFS_WORK_DIRECTORY+"/" + prefix + "/"+train_path
test_path = HDFS_WORK_DIRECTORY + "/" + prefix + "/"+test_path


# In[4]:


EVENT_SQL_FILE = 'sql/event.sql'
PRODUCT_SQL_FILE = 'sql/product.sql'
with open(EVENT_SQL_FILE, 'r') as fd:
    EVENT_SQL = fd.read()
# with open(USER_SQL_FILE, 'r') as fd:
#     USER_INFO_SQL = fd.read()
with open(PRODUCT_SQL_FILE, 'r') as fd:
    PRODUCT_SQL = fd.read()
# print(EVENT_SQL)
# print(PRODUCT_SQL)


# In[5]:


event_col_names = ['user_id', 'user_type', 'member_level', 'time', 'product_id']
product_col_names = [
    'product_id', 'sku_id', 'product_name', 'category_id1', 'category_name1',
    'category_id2', 'category_name2', 'category_id3', 'category_name3',
    'brand_id', 'brand_name'
]
event_path = HDFS_WORK_DIRECTORY+"/" + prefix + "/data/event.parquet"
product_path = HDFS_WORK_DIRECTORY+"/" + prefix + "/data/product.parquet"

print(event_path, product_path)


# In[6]:


df_event = spark.read.format("jdbc").option("driver", "ru.yandex.clickhouse.ClickHouseDriver").option(
    "url", "jdbc:clickhouse://10.254.64.177:8123?zeroDateTimeBehavior=convertToNull").option(
    'query', EVENT_SQL).option("user", "default").load().select(event_col_names)

df_product = spark.read.format("jdbc").option("driver", "com.facebook.presto.jdbc.PrestoDriver").option(
    "url", "jdbc:presto://10.254.8.108:8054/hive/default").option(
    'query', PRODUCT_SQL).option("user", "default").load().select(product_col_names)

df_event.write.parquet(event_path, 'overwrite')
df_product.write.csv(product_path, 'overwrite', header=True)


# In[40]:


df_event = spark.read.parquet(event_path)
df_product = spark.read.csv(product_path, header=True)


# In[41]:


@F.udf(returnType=StringType())
def join_cols(cols):
    return '_'.join([str(c) for c in cols])


df_product = df_product.withColumn('item_id', join_cols(
    F.struct('product_id', 'category_id1', 'category_id2', 'category_id3', 'brand_id')))


# In[42]:


df_u = df_event.groupby('user_id').agg(F.count('product_id').alias('product_cnt'))
df_u = df_u.filter(df_u.product_cnt >= 10)

df_p = df_event.groupby('product_id').agg(F.count('user_id').alias('user_cnt'))
df_p = df_p.filter(df_p.user_cnt >= 10).join(df_product, 'product_id', 'inner')

df_data = df_event.join(df_u, 'user_id', 'inner').join(df_p, 'product_id', 'inner')
col_names = ['user_id', 'user_type', 'member_level', 'time', 'product_id']
df_data = df_data.select(col_names)


# In[43]:


pd_items = df_p.toPandas().sort_values(by='user_cnt', ascending=False).reset_index(drop=True)
pd_users = df_data.select(['user_id', 'user_type', 'member_level']).distinct().toPandas()


# In[44]:


pd_item_id = pd_items['item_id'].drop_duplicates().reset_index().drop(columns='index')
pd_product_id = pd_items['product_id'].drop_duplicates().replace(r'^\s*$', np.nan, regex=True).reset_index().drop(columns='index')
pd_first_classes_id = pd_items['category_id1'].sort_values(ascending=True).drop_duplicates().replace(r'^\s*$', np.nan, regex=True).dropna().reset_index().drop(columns='index')
pd_second_class_id = pd_items['category_id2'].sort_values(ascending=True).drop_duplicates().replace(r'^\s*$', np.nan, regex=True).dropna().reset_index().drop(columns='index')
pd_third_class_id = pd_items['category_id3'].sort_values(ascending=True).drop_duplicates().replace(r'^\s*$', np.nan, regex=True).dropna().reset_index().drop(columns='index')
pd_brand_id = pd_items['item_id'].sort_values(ascending=True).drop_duplicates().replace(r'^\s*$', np.nan, regex=True).dropna().reset_index().drop(columns='index')


pd_user_id = pd_users['user_id'].sort_values(ascending=True).drop_duplicates().replace(r'^\s*$', np.nan, regex=True).dropna().reset_index().drop(columns='index')
pd_user_type = pd_users['user_type'].sort_values(ascending=True).drop_duplicates().replace(r'^\s*$', np.nan, regex=True).dropna().reset_index().drop(columns='index')
pd_member_level = pd_users['member_level'].sort_values(ascending=True).drop_duplicates().replace(r'^\s*$', np.nan, regex=True).dropna().reset_index().drop(columns='index')

pd_item_id.index += 1
pd_product_id.index += 1
pd_first_classes_id.index += 1
pd_second_class_id.index += 1
pd_third_class_id.index += 1
pd_brand_id.index += 1
pd_user_id.index += 1
pd_user_type.index += 1
pd_member_level.index += 1


# In[45]:


pd_items.to_csv(items_info_path, sep=',', index=True, header=True)

pd_item_id.to_csv(vocab_paths['item_id'], index=True, header=False)
pd_product_id.to_csv(vocab_paths['product_id'], index=True, header=False)
pd_first_classes_id.to_csv(vocab_paths['first_class_id'], index=True, header=False)
pd_second_class_id.to_csv(vocab_paths['second_class_id'], index=True, header=False)
pd_third_class_id.to_csv(vocab_paths['third_class_id'], index=True, header=False)
pd_brand_id.to_csv(vocab_paths['brand_id'], index=True, header=False)

pd_user_id.to_csv(vocab_paths['user_id'], index=True, header=False)
pd_user_type.to_csv(vocab_paths['user_type'], index=True, header=False)
pd_member_level.to_csv(vocab_paths['member_level'], index=True, header=False)


# In[46]:


num_vocab = {
    'item_id': len(pd_item_id)+1,
    'product_id': len(pd_product_id)+1,
    'first_class_id': len(pd_first_classes_id)+1,
    'second_class_id': len(pd_second_class_id)+1,
    'third_class_id': len(pd_third_class_id)+1,
    'brand_id': len(pd_brand_id)+1,

    'user_id': len(pd_user_id)+1,
    'user_type': len(pd_user_type)+1,
    'member_level': len(pd_member_level)+1,
}
context_length = 20
min_context_length = 10
neg_sample_num = 5

with open(meta_path, 'wb') as f:
    pickle.dump(vocab_paths, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(num_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump((context_length, neg_sample_num), f, protocol=pickle.HIGHEST_PROTOCOL)


# In[47]:


items_dict = dict(zip(pd_items['product_id'], pd_items['item_id']))


# In[48]:


def approx_zipfian_sampling(num):
    return max(int(np.ceil(np.power(num + 1, random.random()))) - 2, 0)


class UdfPidList(object):
    def __init__(
        self,
        vocab_map: Mapping[Any, Union[str, Any]] = None,
        vocab_list: Sequence[Any] = None,
        vocab_len: int = None,
        ctx_num: int = 4,
        ctx_min_num: int = 0,
        default_value: Union[int, str] = None,
        negative_sample_num: int = 2,
    ):
        """
        Arguments:

            vocab_map: Map[itemid, item_with_info],
                item_id_with_info seperated by underline.
                eg: [itemid]_[classid1]_[classid2]_[classid3]_[brandid]
            vocab_list: items vocabulary (Optional)
            vocab_len: size of items vocabulary (Optional)
            ctx_num: number of context items
            ctx_min_num: minimum number of context items
            negative_sample_num: negative sampled number of item
        """
        self.vocab_map = vocab_map
        self.vocab_list = list(vocab_map.keys()) if self.vocab_map else vocab_list
        self.vocab_len = len(self.vocab_list) if self.vocab_list else vocab_len
        self.ctx_num = ctx_num
        self.ctx_min_num = ctx_min_num
        self.neg_sample_num = negative_sample_num
        self.ctx_seq = np.arange(-ctx_num, 0)
        self.both_ctx_seq = np.arange(-ctx_num, int(ctx_num/2))  # 前面的点击 和 后面一半点击 都不应该筛选出来
        if default_value:
            self.default_value = default_value
        else:
            label_item = self.vocab_map[self.vocab_list[0]] if self.vocab_map else self.vocab_list[0]
            if isinstance(label_item, str) and len(label_item.split('_')) > 1:
                self.default_value = '_'.join(['0']*len(self.vocab_map[self.vocab_list[0]].split('_')))
            elif isinstance(self.vocab_list[0], int):
                self.default_value = 0
            elif isinstance(self.vocab_list[0], float):
                self.default_value = 0
            else:
                self.default_value = '0'

    def __call__(self, tuples: Tuple[Any, str]):
        """
        Arguments:
            tuples: array<time, item_id>

        Returns:
            A tuple 
        """
        items_len = len(tuples)
        if self.ctx_min_num >= items_len:
            return None
        sorted_tuples = sorted(tuples, key=lambda t: t[0], reverse=False)
        items_list = list(map(lambda t: t[1], sorted_tuples))
        dataset = []
        for i in range(self.ctx_min_num, items_len):
            ctx_idx = (i + self.ctx_seq)
            ctx_idx = ctx_idx[np.where((ctx_idx >= 0) & (ctx_idx < items_len))]
            ctx_list = [items_list[j] for j in ctx_idx]

            both_ctx_idx = (i + self.both_ctx_seq)
            both_ctx_idx = both_ctx_idx[np.where((both_ctx_idx >= 0) & (both_ctx_idx < items_len))]
            both_ctx_list = [items_list[j] for j in both_ctx_idx]

            tgt_list = [items_list[i]]
            for _ in range(self.neg_sample_num):
                k = approx_zipfian_sampling(self.vocab_len)
                while self.vocab_list[k] in both_ctx_list + tgt_list:
                    k = approx_zipfian_sampling(self.vocab_len)
                tgt_list.append(self.vocab_list[k])

            default_list = [self.default_value] * (self.ctx_num-len(ctx_list))
            if self.vocab_map:
                dataset.append(([self.vocab_map[c] for c in ctx_list]+default_list,
                                [self.vocab_map[t] for t in tgt_list], len(ctx_list)))
            else:
                dataset.append(ctx_list+default_list, tgt_list, len(ctx_list))

        return dataset


udf_pids = UserDefinedFunction(UdfPidList(vocab_map=items_dict, ctx_num=context_length,
                                          ctx_min_num=min_context_length, default_value='0_0_0_0_0',
                                          negative_sample_num=neg_sample_num),
                               "array<struct<hist_ids:array<string>,label_ids:array<string>,hist_len:integer>>")


# In[49]:


# df_data.filter(df_data.user_id == 4488357).sort(F.asc('time')).select('user_id', 'time', 'product_id').show(150)


# In[50]:


# df_data=df_data.filter(df_data.user_id == 63)
df_d = df_data.groupby('user_id').agg(F.collect_set(F.struct('time', 'product_id')).alias('time_and_product_id'))
df_d = df_d.select('user_id', udf_pids('time_and_product_id').alias('hist_label_rows')).select(
    "user_id", F.explode("hist_label_rows").alias('hist_label')).select(
    "user_id", F.col("hist_label.hist_ids").alias("hist_ids"), F.col("hist_label.label_ids").alias('label_ids'),
    F.col("hist_label.hist_len").alias('hist_len'), F.size(F.col("hist_label.hist_ids")).alias('hist_len2'))
df_u = df_data.select(['user_id', 'user_type', 'member_level']).distinct()

data_df = df_d.join(df_u, 'user_id', 'left').select(
    [F.col('user_id').cast('string').alias('user_id'), 'user_type', 'member_level',
     'hist_ids', 'label_ids', 'hist_len'])
# data_df = data_df.withColumn('user_id_str', F.col('user_id').cast("string"))
# data_df.orderBy(F.rand()).head(20)
# data_df.head(20)


# In[51]:


# df_data.filter(df_data.user_id == 63).sort(F.asc('time')).head(20)
# data_df.head(20)


# In[52]:


train_df, test_df = data_df.randomSplit([0.9, 0.1])


# In[53]:


train_df.repartition(10).write.format("tfrecords").option("recordType", "Example").save(train_path, mode="overwrite")
test_df.repartition(10).write.format("tfrecords").option("recordType", "Example").save(test_path, mode="overwrite")


# In[ ]:


train_df.printSchema()


# In[25]:


# help(train_df.withColumnRenamed)
# help(F.col('user').cast)


# In[ ]:
