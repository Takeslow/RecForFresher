import tensorflow as tf
from model import DeepFM
from utils import create_criteo_dataset, make_datasets
from tensorflow.keras import losses, metrics
from tensorflow.keras.optimizers import Adam
import pandas as pd

if __name__ == '__main__':

    pd.set_option('display.max_columns', 100)
    # TODO：GPU相关设置
    gpus = tf.config.experimental.list_physical_devices('GPU') # 确认 GPU
    for gpu in gpus:   # 按需分配显存
        tf.config.experimental.set_memory_growth(gpu, True)
    #tf.debugging.set_log_device_placement(True)
    # 查看是否运行在GPU上：
    # 若显示：Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0 则说明是运行在GPU上

    # TODO：设定超参数
    test_size=0.1
    batch_size = 4096
    learning_rate = 0.01
    hidden_units = [256, 128, 64]
    output_dim=1
    activation='relu'
    k = 10   # 隐向量v的维度(n,k)
    w_reg = 1e-4  # w的L2正则化系数
    v_reg = 1e-4  # v的L2正则化系数

    # TODO：数据处理
    file_path = 'C:\\Data\\Criteo\\new_test.csv'
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file_path, test_size=test_size)
    #构造 tf.data.Dataset
    train_ds, test_ds = make_datasets(X_train, y_train, X_test, y_test, batch_size=batch_size)

    # TODO：编译模型
    #strategy = tf.distribute.MirroredStrategy()
    # 明确使用 GPU:0 构建分布式策略
    strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0"])
    # MirroredStrategy 会自动在所有可见 GPU 上复制模型，并在每个 batch 内做数据并行
    with strategy.scope():  #在 scope 内构建并编译模型
        model = DeepFM(feature_columns, k, w_reg, v_reg, hidden_units, output_dim, activation)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=losses.BinaryCrossentropy(),
            metrics=[metrics.BinaryAccuracy(), metrics.AUC()]  # ACC与AUC
        )

    # TODO：训练
    epochs = 3
    history = model.fit(
        train_ds,
        #validation_data=test_ds,
        epochs=epochs
    )
    #运行过程中，Keras 会自动在每个 epoch 后输出 loss/acc/auc

    # TODO：评估
    results = model.evaluate(test_ds)
    # model.metrics_names 是 ['loss', 'binary_accuracy', 'auc']
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")
