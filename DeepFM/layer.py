import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense

class FM_layer(Layer):
    def __init__(self, k, w_reg, v_reg):
        super().__init__()
        self.k = k  # 隐向量v的维度(n,k)
        self.w_reg = w_reg  # w的L2正则化系数
        self.v_reg = v_reg  # v的L2正则化系数

    def build(self, input_shape):
        # 全局偏置量w0
        self.w0 = self.add_weight(name='w0', shape=(1,), # 大小为 1,后面会再处理为(batch_size,1)
                                  initializer=tf.zeros_initializer(),
                                  trainable=True,)
        # 一阶线性权重w , 大小为(batch_size,1) , 即(221,1)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg)) #L2正则化
        # 隐向量矩阵v,用于捕捉二阶交互 , 大小为(batch_size,k) , 即 (221,10)
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg)) #L2正则化


    def call(self, inputs, **kwargs):  # inputs的shape:(batchsize,221)
        # TODO：计算一阶线性项
        linear_part = tf.matmul(inputs, self.w) + self.w0   #shape:(batchsize, 1)

        # TODO：计算二阶交互项
        # inter_part1 表示 Σ((vi×xi)²) 注意：是矩阵内各元素自身的2次方
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  #shape:(batchsize, k)

        # inter_part2 表示 Σ(vi²×xi²) 注意：是矩阵内各元素自身的2次方
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))  #shape:(batchsize, k)

        # inter_part 表示 0.5× Σ { Σ((vi×xi)²) -  Σ(vi²×xi²)  }
        inter_part = 0.5*tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True) #shape:(batchsize, 1)
        # 按隐向量维度求和：tf.reduce_sum(…, axis=-1) 将每个样本在k维度上的值相加，结果的shape：(batchsize,1)

        # TODO：一阶线性项与二阶交互项相加,作为输出
        output = linear_part + inter_part

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



