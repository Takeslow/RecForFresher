from layer import Deep_layer,Cross_layer

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding,Dense


class DCN(Model):
    def __init__(self, feature_columns, hidden_units, output_dim, activation, layer_num, reg_w, reg_b):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embedding_layer = {'embed_layer' + str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                for i, feat in enumerate(self.sparse_feature_columns)}
        self.dense_layer = Deep_layer(hidden_units, output_dim, activation)
        self.cross_layer = Cross_layer(layer_num, reg_w=reg_w, reg_b=reg_b)
        self.output_layer = Dense(1, activation=None)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:39]
        # embedding
        sparse_embed = tf.concat([self.embedding_layer['embed_layer' + str(i)](sparse_inputs[:, i]) \
                                  for i in range(sparse_inputs.shape[-1])], axis=-1)

        # Cross_layer与Deep_layer的输入均为x , 即稠密特征与embedding后的稀疏特征的拼接
        x = tf.concat([dense_inputs, sparse_embed], axis=1)

        # Cross_layer
        cross_output = self.cross_layer(x)
        # Deep_layer
        dnn_output = self.dense_layer(x)

        x = tf.concat([cross_output, dnn_output], axis=1)
        output = tf.nn.sigmoid(self.output_layer(x)) # 再通过一个MLP层,并进行sigmoid
        return output