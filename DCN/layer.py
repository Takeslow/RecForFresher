import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense


class Cross_layer(Layer):
    def __init__(self, layer_num, reg_w, reg_b):
        super().__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        self.cross_weight = [
            self.add_weight(name='w'+str(i),
                            shape=(input_shape[1], 1),  # 每层权重wl的形状
                            initializer=tf.random_normal_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.reg_w), # 对每层的w进行L2正则化
                            trainable=True)
            for i in range(self.layer_num)]  # 共创建layer_num层
        self.cross_bias = [
            self.add_weight(name='b'+str(i),
                            shape=(input_shape[1], 1), # 每层偏置bl的形状
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.reg_b), # 对每层的b进行L2正则化
                            trainable=True)
            for i in range(self.layer_num)]  # 共创建layer_num层

    def call(self, inputs, **kwargs):
        # 在第2维添加大小为1的轴，将形状从 (batch_size, dim) 变为 (batch_size, dim, 1)，便于后续矩阵运算
        # x0:常量，始终保留原始输入，用于所有交叉层的运算
        x0 = tf.expand_dims(inputs, axis=2)   # (batch_size, dim, 1)
        # xl:第l层的输入，初始化为x0
        xl = x0  # (batch_size, dim, 1)
        for i in range(self.layer_num):
            # 先乘后两项（忽略第一维，(dim, 1)表示一个样本的特征）
            # tf.transpose(xl, [0, 2, 1]) : 将xl从 (batch, dim, 1) 转为 (batch, 1, dim)
            # tf.matmul: 将转换了维度的 xl 与权重矩阵 (dim,1) 相乘，得到 (batch,1,1)，即标量wi×xl
            xl_w = tf.matmul(tf.transpose(xl, [0, 2, 1]), self.cross_weight[i]) # (batch_size, 1, 1)
            # 乘x0，再加上bl、xl
            xl = tf.matmul(x0, xl_w) + self.cross_bias[i] + xl  # (batch_size, dim, 1)

        output = tf.squeeze(xl, axis=2)  # (batch_size, dim)
        return output

class Deep_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation):
        super().__init__()
        self.hidden_layer = [Dense(i, activation=activation) for i in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output