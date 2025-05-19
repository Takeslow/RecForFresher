from layer import Deep_layer,FM_layer

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding


class DeepFM(Model):
    def __init__(self, feature_columns, k, w_reg, v_reg, hidden_units, output_dim, activation):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embedding_layer = {'embed_layer' + str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                for i, feat in enumerate(self.sparse_feature_columns)}
        self.FM = FM_layer(k, w_reg, v_reg)  #FM部分
        self.Deep = Deep_layer(hidden_units, output_dim, activation)  #Deep部分

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:39]
        # embedding处理    sparse_embed的shape:(batch_size,208) 208=26×8
        sparse_embed = tf.concat([self.embedding_layer['embed_layer' + str(i)](sparse_inputs[:, i]) \
                                  for i in range(sparse_inputs.shape[-1])], axis=-1)

        x = tf.concat([dense_inputs, sparse_embed], axis=-1) #shape:(batch_size,221) 221=208+13

        fm_output = self.FM(x)  # shape:(batch_size, 1)
        deep_output = self.Deep(x) # shape:(batch_size, 1)

        output = tf.nn.sigmoid(0.5 * (fm_output + deep_output))
        return output

