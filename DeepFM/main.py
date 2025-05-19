import tensorflow as tf
from model import DeepFM
from utils import create_criteo_dataset, make_datasets
from tensorflow.keras import losses, metrics
from tensorflow.keras.optimizers import Adam


if __name__ == '__main__':

    # TODO：GPU相关设置 (PS:根据自身条件设置)
    gpus = tf.config.experimental.list_physical_devices('GPU') # 确认 GPU
    for gpu in gpus:   # 按需分配显存
        tf.config.experimental.set_memory_growth(gpu, True)

    # TODO：设定超参数
    test_size=0.1 # 用于划分训练集与测试集(0.1表示9:1)
    batch_size = 4096
    epochs = 3
    learning_rate = 0.01
    hidden_units = [256, 128, 64]
    output_dim=1
    activation='relu'
    k = 10   # 隐向量v的维度(n,k)
    w_reg = 1e-4  # w的L2正则化系数
    v_reg = 1e-4  # v的L2正则化系数

    # TODO：数据处理
    file_path = 'D:\\Data\\Criteo\\criteo_small.csv'
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file_path, test_size=test_size)
    #构造 tf.data.Dataset
    train_ds, test_ds = make_datasets(X_train, y_train, X_test, y_test, batch_size=batch_size)

    # TODO：编译模型
    # 明确使用 GPU:0 构建分布式策略 PS:根据自身条件设置不同策略
    strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0"])
    # MirroredStrategy 会自动在所有可见 GPU 上复制模型，并在每个 batch 内做数据并行
    with strategy.scope():  #在 scope 内构建并编译模型
        model = DeepFM(feature_columns, k, w_reg, v_reg, hidden_units, output_dim, activation)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate), #Adam优化
            loss=losses.BinaryCrossentropy(), #交叉熵
            metrics=[metrics.BinaryAccuracy(), metrics.AUC()]  # ACC与AUC
        )

    # TODO：训练
    history = model.fit(
        train_ds,
        epochs=epochs
    )
    #运行过程中，Keras 会自动在每个 epoch 后输出 loss/acc/auc

    # TODO：评估
    results = model.evaluate(test_ds)
    # model.metrics_names 是 ['loss', 'binary_accuracy', 'auc']
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")
