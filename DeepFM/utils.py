import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}


def create_criteo_dataset(file_path, embed_dim=8, test_size=0.2):
    data = pd.read_csv(file_path) #数据结构 [ label I1 I2... I13 C1 C2 ... C26 ]
    #data.drop(data.index[10000:4587166], inplace=True)
    dense_features = ['I' + str(i) for i in range(1, 14)]  #稠密特征I1-I13
    sparse_features = ['C' + str(i) for i in range(1, 27)]  #稀疏特征C1-C26

    # 缺失值填充
    data[dense_features] = data[dense_features].fillna(0) #连续值缺失时填0
    data[sparse_features] = data[sparse_features].fillna('-1')  #提供一个专属类别“-1”，代表“缺失”

    # 归一化
    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])
    # MinMaxScaler的fit函数可以计算出某列特征中的最大值、最小值、均值、方差等属性，与transform函数配套使用；
    # transform函数可以对某列特征进行归一化和标准化；
    # fit_transform函数对数据先拟合
    # fit，找到数据的整体指标，如均值、方差、最大值最小值等，然后对数据集进行转换transform，从而实现数据的标准化、归一化操作。
    # （参考链接https: // cloud.tencent.com / developer / article / 1770568）

    # LabelEncoding编码(deep侧输入)
    for col in sparse_features:
        data[col] = LabelEncoder().fit_transform(data[col])
    # Label Encoder将一列文本数据转化成0-N，One hot Encoder将一列文本数据转化成一列或多列只有0和1的数据；

    # Onehot编码(wide侧输入)
    sparse_df = data[sparse_features].astype(str)  # 把 C1–C26 全部转为字符串，即仅对稀疏特征
    onehot_data = pd.get_dummies(sparse_df, prefix=sparse_features)
    # get_dummies函数是利用pandas实现one hot encode的方式；

    # 准备训练输入 X 和标签 y  df=df.loc[:,[‘name1’,‘name2’,‘name3’]]
    X = data.drop('label',axis=1)
    y = data[['label']]

    # 特征拼接
    X = pd.concat([X, onehot_data], axis=1)  #(数据总量,36323) 36323：I1-I13 C1-C26 C1_0-C26_xxx

    # feature_columns是二维list，list[0]有13个特征，list[1]有26个特征
    feature_columns = [
        [denseFeature(f) for f in dense_features],
        [sparseFeature(f, int(data[f].nunique()), embed_dim)
         for f in sparse_features]
    ]

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return feature_columns, (X_train, y_train), (X_test, y_test)


# 用于打乱、预取数据,同时生成Dataset型数据，方便训练使用
def make_datasets(X_train, y_train, X_test, y_test,
                  batch_size=1024, shuffle_buffer=100000):
    #shuffle_buffer: shuffle 时的缓冲区大小，越大越能保证充分随机，但占用更多内存

    # 将输入元组 (features, labels) “逐行”切片成若干元素，每个元素是 (X_train[i], y_train[i])
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # 打乱、批处理、预取
    train_ds = (train_ds
                .shuffle(shuffle_buffer, reshuffle_each_iteration=True)
                .batch(batch_size)
                .prefetch(AUTOTUNE))  # 使用 experimental.AUTOTUNE
    #.shuffle(buffer_size, reshuffle_each_iteration) :
    # buffer_size：内部维护一个大小为buffer_size的环形队列，从队列中随机抽取输出元素。
    # reshuffle_each_iteration = True：每个epoch均重新打乱，增强随机性；设为False则固定一次顺序。
    # buffer_size应 ≥ 数据集总大小，才能实现完全随机；若受内存限制，可取数据量的10–100倍折衷

    #.prefetch(AUTOTUNE) :
    # AUTOTUNE = tf.data.experimental.AUTOTUNE，让TensorFlow自动选择预取大小。
    # prefetch(n)会在模型训练时异步准备未来n个batch，缓解数据读取与训练的串行等待，提升GPU利用率

    # 同训练集的处理
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, test_ds