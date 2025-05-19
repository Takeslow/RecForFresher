import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense

class Wide_layer(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        #偏置值w0
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(), #初始化为0
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(), #随机生成正态值
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(1e-4)) #L2正则化

    def call(self, inputs, **kwargs):   #输入（inputs）为传入的wide_inputs
        x = tf.matmul(inputs, self.w) + self.w0     #x的shape: (batchsize, 1)
        #w的shape:(xxx×1) w0的shape:(1,),即w0是长度为1的一维张量
        #TensorFlow 支持 NumPy 风格的自动广播
        #在tf.matmul(inputs, self.w)得到的张量形状是(batch_size, 1)
        #w0视为(1×1),与(batch_size,1)比较时第二维相同，第一维可因值为1而沿批次方向复制batch_size次

        return x

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