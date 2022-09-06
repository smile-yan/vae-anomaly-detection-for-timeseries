"""
anomaly detection by vae for time series
by smileyan (root@smileyan.cn)
"""
import tensorflow as tf
import numpy as np
import time


def reparameterize(mu, log_var):
    """重参数化，计算隐变量 z = μ + ε ⋅ σ
    :param mu:  均值
    :param log_var: 方差的 log 值
    :return: 隐变量 z
    """
    # log σ^2 -> σ
    std = tf.exp(log_var * 0.5)
    eps = tf.random.normal(std.shape)
    return mu + eps * std


def log_normal_pdf(sample, mean, log_var, axis=1):
    """ 求解正态分布(mean, var) 中的概率密度
    :param sample: 待求样本
    :param mean: 分布的均值
    :param log_var: 分布的方差的对数值
    :param axis: reduce_sum 的参数
    :return: 概率密度的对数值
    """
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-log_var) + log_var + log2pi), axis=axis)


class BasicVAE(tf.keras.Model):
    def __init__(self, latent_size=4, data_shape=120):
        super(BasicVAE, self).__init__()
        # input => h
        self.fc1 = tf.keras.layers.Dense(100)
        # h => μ and log σ^2
        self.fc2 = tf.keras.layers.Dense(latent_size)
        self.fc3 = tf.keras.layers.Dense(latent_size)

        # sampled z => h
        self.fc4 = tf.keras.layers.Dense(100)
        # h => original data
        self.fc5 = tf.keras.layers.Dense(data_shape)

    def encode(self, x):
        """encode过程，返回 μ 和 log σ^2
        :param x: 单窗口数据
        :return:  μ 和 log σ^2
        """
        h = tf.nn.relu(self.fc1(x))
        # mu, log_variance
        return self.fc2(h), self.fc3(h)

    def decode_logits(self, z):
        h = tf.nn.relu(self.fc4(z))
        return self.fc5(h)

    def decode(self, z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def reconstruction_prob(self, x, sample_times=10):
        """计算本次数据的重构概率 E_{q_φ(z|x)}[log p_θ (x|z)]
        :param x: 测试数据
        :param sample_times: 采样时间
        :return: 重构概率
        """
        # tf.reshape(x, [-1, 120])
        mean, log_var = self.encode(x)
        samples_z = []
        for i in range(sample_times):
            z = reparameterize(mean, log_var)
            samples_z.append(z)
        reconstruction_prob = []
        for z in samples_z:
            x_logit = self.decode_logits(z)
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
            # only take the last point's reconstruction probability
            reconstruction_prob.append(cross_ent[0][-1])

        return tf.reduce_mean(reconstruction_prob, axis=-1)

    def compute_loss(self, x):
        """loss function, i.e., ELBO
           :param x: one batch of validation set
           :return: elbo
       """
        mean, log_var = self.encode(x)
        z = reparameterize(mean, log_var)
        x_logit = self.decode_logits(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        log_p_x_z = -tf.reduce_sum(cross_ent)
        log_p_z = log_normal_pdf(z, 0., 0.)
        log_q_z_x = log_normal_pdf(z, mean, log_var)
        return -tf.reduce_mean(log_p_x_z + log_p_z - log_q_z_x)

    def fit(self, dataset_x, train_epochs=10, batch_size=100, optimizer=tf.keras.optimizers.Adam(1e-3), train_rate=0.7):
        """train vae model
        :param dataset_x: window based data
        :param batch_size: size of each batch default 10
        :param train_rate: training data rate default 0.7
        :param train_epochs: training times default 100
        :param optimizer: one optimizer default Adam
        """
        data_split = int(len(dataset_x) * train_rate)
        train_data = tf.cast(dataset_x[:data_split], tf.float32)
        test_data = tf.cast(dataset_x[data_split:], tf.float32)

        # num_batches = data_split // batch_size
        training_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)

        for epoch in range(train_epochs):
            start_time = time.time()
            for step, x in enumerate(training_dataset):
                x = tf.reshape(x, [-1, 120])
                with tf.GradientTape() as tape:
                    loss = self.compute_loss(x)
                gradients = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            train_end_time = time.time()
            loss_mean = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                test_x = tf.reshape(test_x, [-1, 120])
                loss_mean(self.compute_loss(test_x))
            elbo = -loss_mean.result()
            test_end_time = time.time()
            print('Epoch: {}/{}, test set ELBO: {:.4f}, train time elapse : {:.2f} s, test time elapse : {:.2f} s'
                  .format(epoch+1, train_epochs, elbo, train_end_time - start_time, test_end_time - train_end_time))
