from layer import Wide_layer, Deep_layer

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding

class WideDeep(Model):
    def __init__(self, feature_columns, hidden_units, output_dim, activation):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        #为每个稀疏特征实例化一个Embedding矩阵
        self.embedding_layer = {'embed_layer'+str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                for i,feat in enumerate(self.sparse_feature_columns)}
        self.wide = Wide_layer()
        self.deep = Deep_layer(hidden_units, output_dim, activation)

    def call(self, inputs):
        # dense_inputs: 数值特征，13维
        # sparse_inputs： 类别特征，26维
        # onehot_inputs：onehot处理的类别特征(wide侧的输入)
        dense_inputs, sparse_inputs, onehot_inputs = inputs[:, :13], inputs[:, 13:39], inputs[:, 39:]

        # wide部分
        wide_input = tf.concat([dense_inputs, onehot_inputs], axis=-1)  # 稠密特征＋onehot编码的稀疏特征
        #axis=0表示按第1个维度拼接，axis=1表示按第二个维度拼接，axis=-1表示按倒数第一个维度拼接
        wide_output = self.wide(wide_input)  #得到形状(batch_size,1)的输出wide_output
        #wide部分要包含所有一阶特征（包括连续和类别，即dense_inputs和onehot_inputs）

        # deep部分
        # PS：spares_inputs实为batch_size×26
        # PS：spares_embed实为batch_size×208(208=8×26，即embed_size×26)
        sparse_embed = tf.concat([self.embedding_layer['embed_layer'+str(i)](sparse_inputs[:, i]) \
                        for i in range(sparse_inputs.shape[-1])], axis=-1)
        deep_input = tf.concat([dense_inputs, sparse_embed], axis=-1)
        deep_output = self.deep(deep_input)

        output = tf.nn.sigmoid(0.5*(wide_output + deep_output))
        return output