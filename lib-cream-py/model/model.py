import os

import tensorflow as tf
from keras import Model, Input
from keras.src import optimizers, layers
from keras.src.layers import Lambda
from keras.src.saving import load_model

from .contextual_block import ContextualBlock
from .decoder import Decoder
from .disciminator_red import DiscriminatorRed
from .encoder import Encoder

class InpaintNN:
    def __init__(self, model_path: str, input_height=256, input_width=256, batch_size=1, create_model=False):
        super(InpaintNN, self).__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self.model_path = model_path
        self.create_model = create_model
        if not create_model:
            self.check_model_file()
            self.model = load_model(self.model_path)
        else:
            self.model = self.build_model()
            if os.path.exists(self.model_path):
                self.load_checkpoint()

    def check_model_file(self):
        if not os.path.exists(self.model_path):
            print("\nMissing Model, download model")
            print("Read: https://github.com/deeppomf/DeepCreamPy/blob/master/docs/INSTALLATION.md#run-code-yourself \n")
            exit(-1)

    def build_model(self):
        X = Input(shape=(self.input_height, self.input_width, 3), batch_size=self.batch_size, dtype=tf.float32)
        Y = Input(shape=(self.input_height, self.input_width, 3), batch_size=self.batch_size, dtype=tf.float32)
        MASK = Input(shape=(self.input_height, self.input_width, 3), batch_size=self.batch_size, dtype=tf.float32)
        # todo: unclear
        IT = 0

        # Model layers
        input_tensor = layers.Concatenate(axis=-1)([X, MASK])

        vec_en = Encoder(name='G_en')(input_tensor)
        vec_con = ContextualBlock(name='CB1', k_size=3, lamda=50.0, stride=1)((vec_en, vec_en, MASK))

        decoder = Decoder(name='G_de', size1=self.input_height, size2=self.input_height)
        I_co = decoder(vec_en)
        I_ge = decoder(vec_con)

        image_result = I_ge * (1 - MASK) + Y * MASK

        # Discriminator
        disc_red = DiscriminatorRed(name='disc_red')
        D_real_red = disc_red(Y)
        D_fake_red = disc_red(image_result)

        # Losses
        loss_D = Lambda(
            lambda t: tf.reduce_mean(tf.nn.relu(1 + t[0])) + tf.reduce_mean(tf.nn.relu(1 - t[1]))
        )([D_fake_red, D_real_red])

        loss_GAN = Lambda(lambda t: -tf.reduce_mean(t))(D_fake_red)

        loss_s_re = Lambda(lambda t: tf.reduce_mean(tf.abs(t[0] - t[1])))([I_ge, Y])

        loss_hat = Lambda(lambda t: tf.reduce_mean(tf.abs(t[0] - t[1])))([I_co, Y])

        # todo never used?
        # SSIM (Structural Similarity Index)
        # A = tf.image.rgb_to_yuv((image_result + 1) / 2.0)
        # A_Y = tf.cast(A[:, :, :, 0:1] * 255.0, tf.int32)
        # B = tf.image.rgb_to_yuv((X + 1) / 2.0)
        # B_Y = tf.cast(B[:, :, :, 0:1] * 255.0, tf.int32)
        # ssim = tf.reduce_mean(tf.image.ssim(tf.cast(A_Y, tf.float32), tf.cast(B_Y, tf.float32), max_val=255.0))

        alpha = IT / 1000000
        # todo loss_D, loss_G and optimizer_D, optimizer_G
        # var_D = [v for v in tf.global_variables() if v.name.startswith('disc_red')]
        # var_G = [v for v in tf.global_variables() if v.name.startswith('G_en') or v.name.startswith('G_de') or v.name.startswith('CB1')]
        loss_G = 0.1 * loss_GAN + 10 * loss_s_re + 5 * (1 - alpha) * loss_hat

        # var_D = [v for v in tf.global_variables() if v.name.startswith('d')]
        # var_G = [v for v in tf.global_variables() if v.name.startswith('g') or v.name.starts_with('h')]

        # Optimizers
        optimizer_D = optimizers.Adam(learning_rate=0.0004, beta_1=0.5,
                                            beta_2=0.9)  # .minimize(Loss_D, var_list=var_D)
        optimizer_G = optimizers.Adam(learning_rate=0.0001, beta_1=0.5,
                                            beta_2=0.9)  # .minimize(Loss_G, var_list=var_G)

        model = Model(inputs=[X, Y, MASK], outputs=image_result)
        model.compile()
        if os.path.exists(self.model_path):
            self.load_checkpoint()
        return model

    def load_checkpoint(self):
        checkpoint_path = tf.train.latest_checkpoint(self.model_path)

        if checkpoint_path:
            self.model.load_weights(checkpoint_path)
            print(f"Model restored from {checkpoint_path}")
        else:
            print("\nNo checkpoint found")
            exit(-1)

    def predict_image(self, censored, unused, mask):
        return self.model(censored, unused, mask, training=False)


if __name__ == "__main__":
    m = InpaintNN("", create_model=True)
    m.model.summary(expand_nested=True)
