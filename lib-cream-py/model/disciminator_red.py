import tensorflow as tf
from keras import Layer, Model
from keras.src.initializers import TruncatedNormal


class SNConv2D(Layer):
    def __init__(self, output_dim, kernel_size, stride, name: str):
        super(SNConv2D, self).__init__()
        self.u = None
        self.stride = stride
        self.b = None
        self.w = None
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.name = name

    def build(self, input_shape):
        c = input_shape[3]

        self.w = self.add_weight(
            name=self.name + 'w',
            shape=(self.kernel_size, self.kernel_size, c, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.b = self.add_weight(
            name=self.name + 'b',
            shape=(self.output_dim,),
            initializer=tf.zeros_initializer(),
            trainable=True,
        )

        self.u = self.add_weight(
            name=self.name + 'w',
            shape=[1, self.w.shape[-1]], initializer=TruncatedNormal,
            trainable=False
        )

    def call(self, inputs):
        return tf.nn.conv2d(inputs, filters=spectral_norm(self.w, self.u), strides=[1, self.stride, self.stride, 1],
                            padding='SAME') + self.b


class DiscriminatorRed(Model):
    def __init__(self, name=str):
        super(DiscriminatorRed, self).__init__(name=name)

        self.conv1 = SNConv2D(64, 5, 2, 'l1')
        self.conv2 = SNConv2D(128, 5, 2, 'l2')
        self.conv3 = SNConv2D(256, 5, 2, 'l3')
        self.conv4 = SNConv2D(256, 5, 2, 'l4')
        self.conv5 = SNConv2D(256, 5, 2, 'l5')
        self.conv6 = SNConv2D(512, 5, 2, 'l6')
        self.dense = DenseRedSN('l7')

    def call(self, inputs):
        x = self.conv1(inputs)
        # L1 = instance_norm(L1, 'di1')
        x = tf.nn.leaky_relu(x)

        x = self.conv2(x)
        # L2 = instance_norm(L2, 'di2')
        x = tf.nn.leaky_relu(x)

        x = self.conv3(x)
        # L3 = instance_norm(L3, 'di3')
        x = tf.nn.leaky_relu(x)

        x = self.conv4(x)
        # L4 = instance_norm(L4, 'di4')
        x = tf.nn.leaky_relu(x)

        x = self.conv5(x)
        # L5 = instance_norm(L5, 'di5')
        x = tf.nn.leaky_relu(x)

        x = self.conv6(x)
        # L6 = instance_norm(L6, 'di6')
        x = tf.nn.leaky_relu(x)

        x = self.dense(x)

        return x


class DenseRedSN(Layer):
    def __init__(self, name: str):
        super(DenseRedSN, self).__init__()
        self.u = None
        self.bias = None
        self.weight = None
        self.name = name

    def build(self, input_shape):
        h, w, c = input_shape[1], input_shape[2], input_shape[3]

        self.weight = self.add_weight(
            name=self.name + "_w",
            shape=[h * w, 1, c, 1],
            initializer="glorot_uniform",
            trainable=True
        )

        self.bias = self.add_weight(
            name=self.name + "_b",
            shape=[1, h, w, 1],
            initializer="zeros",
            trainable=True
        )
        self.u = [
            self.add_weight(name=f"{self.name}w_{i}u", shape=[1, self.weight.shape[-1]], initializer=TruncatedNormal,
                            trainable=False)
            for i in range(h * w)
        ]

    def call(self, inputs):
        h, w, c = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        sn_w_list = [spectral_norm(self.weight[i: i + 1, :, :, :], self.u[i])
                     for i in range(h * w)
                     ]
        sn_w = tf.concat(sn_w_list, axis=0)
        w_rs = tf.reshape(sn_w, [h, w, c, 1])
        w_rs_t = tf.transpose(w_rs, [3, 0, 1, 2])

        output_red = tf.reduce_sum(inputs * w_rs_t + self.bias, axis=3, keepdims=True)
        return output_red


def spectral_norm(w, u, iteration=1):
    """Applies spectral normalization to weight tensor w."""
    w_reshaped = tf.reshape(w, [-1, w.shape[-1]])
    u_hat = u
    v_hat = None
    for _ in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, w_reshaped, transpose_b=True)
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w_reshaped)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), u_hat, transpose_b=True)
    w_norm = w_reshaped / sigma

    u.assign(u_hat)
    return tf.reshape(w_norm, w.shape)


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)
