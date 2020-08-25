import tensorflow as tf
from tensorflow.keras import models, layers


# See too https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
class SiameseModel:

    def __init__(self, base_network=None):
        self.base_network = base_network
        if self.base_network is None:
            self.base_network = self.create_base_network()

        self.model = self.create_siamese_model()
        self.input_shape = (224, 224)

    @staticmethod
    def create_base_network():
        # load resnet as base model
        input_shape = (224, 224, 3)
        resnet = tf.keras.applications.resnet50.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
        )
        last_layer = resnet.get_layer('conv5_block3_out')
        conv_model = models.Model(
            inputs=resnet.input,
            outputs=last_layer.output
        )

        # freeze resnet layers
        resnet.trainable = False

        # open conf layer blocks4 and 5
        for layer in conv_model.layers:
            if 'conv5_block' in layer.name:
                layer.trainable = True
            if 'conv4_block' in layer.name:
                layer.trainable = True

        # add new layers to create an embedding
        inputs = tf.keras.Input(shape=(input_shape))
        x = conv_model(inputs, training=True)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # embedding vector of size 128
        outputs = tf.keras.layers.Dense(128)(x)
        base_model = tf.keras.Model(inputs, outputs)

        return base_model

    def create_siamese_model(self):
        input_shape = (
            self.base_network.input_shape[1],
            self.base_network.input_shape[2],
            self.base_network.input_shape[3]
        )

        input_left = layers.Input(shape=input_shape)
        input_right = layers.Input(shape=input_shape)

        # the weights of the base_network will be shared
        embedding_left = self.base_network(input_left)
        embedding_right = self.base_network(input_right)

        l1_layer = layers.Lambda(lambda pair: tf.math.abs(pair[0] - pair[1]))
        l1_distance = l1_layer([embedding_left, embedding_right])

        drop = layers.Dropout(0.5)(l1_distance)
        prediction = layers.Dense(1, activation='sigmoid')(drop)

        return models.Model(inputs=[input_left, input_right], outputs=prediction)
