#!/usr/bin/env python
# coding: utf-8

# ### Siamese Network
#
# Experiment to use a siamese network like outlined in
# https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d or
# https://github.com/aspamers/siamese or
# https://mc.ai/advance-ai-face-recognition-using-siamese-networks/
#
import tensorflow as tf

from datetime import datetime as datetime
from sklearn.model_selection import train_test_split
import os
import json

from cnn_helper import load_json_as_df
from siamese_util import cls_sample_pairs, create_pairs_all_classes
from image_pair_generator import ImagePairGenerator
from siamese_model import SiameseModel

import siamese_config as config

# paths
project_dir = config.project_dir
data_dir = config.data_dir
images_dir = config.images_dir

# load image data into dataframe
tdf = load_json_as_df(data_dir, 'mri-images')

# configs

train_config = {
    'batch_size': 32,
    'image_size': (224, 224),
    'input_shape': (224, 224, 3),
    'train_samples_per_class': 120,
    'validation_samples_per_class': 30,
    'test_size': 0.2,
    'epochs': 120,
    'seed': 47,
    'col': 'sequence'
}
# configs
"""
batch_size = 32
image_size = (224, 224)
input_shape = (image_size[0], image_size[1], 3)
train_samples_per_class = 120
validation_samples_per_class = 30
test_size = 0.2

epochs = 120
seed = 47
"""
# define classes, either the perspective classes, or the sequence classes or the combination of them

# col = 'perspective'
# col = 'sequence'
# col = 'perspective-sequence'

# image augmentation configuration
# rescale does not work with random_transform -> rescale image after random_transform
imagegen_parameters = dict(
    rescale=1. / 255,
    rotation_range=15.0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.005,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest',
)

# end config

# Prepare train and test data
def create_train_test_data(train_config, tdf):

    col = train_config['col']

    train_df, validation_df = train_test_split(tdf, test_size=train_config['test_size'], stratify=tdf[train_config['col']])
    # train_df, validation_df = train_test_split(tdf, test_size=0.2)

    trainclasses_df = train_df.groupby(by=[train_config['col']])
    # available_classes = set(classes_df.groups.keys())
    print(f'Number of available classes: {len(set(trainclasses_df.groups.keys()))}')
    p = [print(f'Class: {key}\t Size: {len(trainclasses_df.get_group(key))}') for key in set(trainclasses_df.groups.keys())]

    train_pairs, train_labels = create_pairs_all_classes(
        train_df,
        col,
        n_samples_per_class=train_config['train_samples_per_class'],
        random_state=train_config['seed'])

    validation_pairs, validation_labels = create_pairs_all_classes(
        validation_df,
        col,
        n_samples_per_class=train_config['validation_samples_per_class'],
        random_state=train_config['seed'])

    # create image generator with image augmentation
    generator = ImagePairGenerator(imagegen_parameters)
    pairs_batch_gen = generator.generate_image_pairs(
        train_pairs,
        train_labels,
        train_config['batch_size'],
        train_config['image_size'],
        images_dir
    )

    validation_pairs_batch_gen = generator.generate_image_pairs(
        validation_pairs,
        validation_labels,
        train_config['batch_size'],
        train_config['image_size'],
        images_dir
    )

    return pairs_batch_gen, train_labels, validation_pairs_batch_gen, validation_labels


def train(train_config, siam_model, pairs_batch_gen, validation_pairs_batch_gen):

    logs = os.path.join(data_dir, 'logs', f'siamese_network_{train_config["col"]}', datetime.now().strftime("%Y%m%d-%H%M%S"))

    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs,
        histogram_freq=1,
        profile_batch='500,520'
    )

    # rms = tf.keras.optimizers.RMSprop()
    adam = tf.keras.optimizers.Adam(learning_rate=0.003)
    siam_model.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    logs = os.path.join(config.src_dir, 'logs', f'siamese_network_{train_config["col"]}', datetime.now().strftime("%Y%m%d-%H%M%S"))

    siamese_checkpoint_path = './siamese_checkpoint.hdf5'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        siamese_checkpoint_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True
    )
    siamese_callbacks = [
        # EarlyStopping(monitor='accuracy', min_delta=0.01, patience=10, verbose=0),
        checkpoint,
        # tboard_callback
    ]

    siam_history = siam_model.model.fit(
        pairs_batch_gen,
        validation_data=validation_pairs_batch_gen,
        steps_per_epoch=len(train_labels) // train_config['batch_size'],
        validation_steps=len(validation_labels) // train_config['batch_size'],
        #callbacks=siamese_callbacks,
        epochs=train_config['epochs']
    )

    date_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    siam_model.model.save(f'resnet50_siamese_{train_config["col"]}_{date_now}.h5')
    siam_model.base_network.save(f'resnet50_siamese_base_{train_config["col"]}_{date_now}.h5')

    # Get the dictionary containing each metric and the loss for each epoch
    history_dict = siam_history.history
    # Save it under the form of a json file
    json.dump(history_dict, open(f'siam_history_{date_now}.txt', 'w'))

    siam_model.model.save(f'resnet50_siamese_{train_config["col"]}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.h5')

    print('Training finished')


if __name__ == "__main__":

    #config
    train_config = {
        'batch_size': 32,
        'image_size': (224, 224),
        'input_shape': (224, 224, 3),
        'train_samples_per_class': 120,
        'validation_samples_per_class': 30,
        'test_size': 0.2,
        'epochs': 120,
        'seed': 47,
        'col': 'sequence'
    }
    # the model
    siam_model = SiameseModel()

    siam_model.base_network.summary()
    siam_model.model.summary()

    tdf = load_json_as_df(data_dir, 'mri-images')

    pairs_batch_gen, train_labels, validation_pairs_batch_gen, validation_labels = create_train_test_data(train_config, tdf)

    train(train_config, siam_model, pairs_batch_gen, validation_pairs_batch_gen)