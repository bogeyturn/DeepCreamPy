import tensorflow as tf

from keras import Layer, Model
from keras.src.layers import Conv2D


class ConvNN(Layer):
    def __init__(self, dims1, dims2, size1, size2, k_size=3):
        super(ConvNN, self).__init__()
        self.dims1 = dims1
        self.dims2 = dims2
        self.size1 = size1
        self.size2 = size2
        self.k_size = k_size

        # Define layers
        self.conv1 = Conv2D(dims1, (k_size, k_size), strides=(1, 1), padding='valid', activation=None)
        self.conv2 = Conv2D(dims2, (k_size, k_size), strides=(1, 1), padding='valid', activation=None)

    def call(self, inputs, training=False):
        print(inputs)
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        x = self.conv1(x)
        x = tf.nn.elu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        x = self.conv2(x)
        x = tf.nn.elu(x)

        # Resize output
        x = tf.image.resize(x, (self.size1, self.size2), method='nearest')
        return x


class Decoder(Model):
    def __init__(self, size1, size2, name: str):
        super(Decoder, self).__init__(name=name)
        self.size1 = size1
        self.size2 = size2
        self.name = name
        self.dl1 = ConvNN(128, 128, int(size1 // 4), int(size2 // 4))
        self.dl2 = ConvNN(64, 64, int(size1 // 2), int(size2 // 2))
        self.dl3 = ConvNN(32, 32, int(size1), int(size2))
        self.dl4 = ConvNN(16, 16, int(size1), int(size2))
        self.final_conv = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=None)

    def call(self, inputs, training=False):
        print(inputs)
        x = self.dl1(inputs)
        x = self.dl2(x)
        x = self.dl3(x)
        x = self.dl4(x)
        x = self.final_conv(x)
        x = tf.clip_by_value(x, -1.0, 1.0)
        return x
