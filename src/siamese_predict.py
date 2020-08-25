import tensorflow as tf

import cv2
import numpy as np
import sys
from tensorflow import keras
import ntpath

import json

from cnn_helper import *
from helper import *

import siamese_config as config


class SiamesePredictor:

    def __init__(self, config, dataframe, siam_model, embedding_model):
        self.config = config
        self.model = siam_model
        self.embedding_model = embedding_model
        self.dataframe = dataframe
        self.image_loader = ImageLoader(config.preprocess_input)
        self.jury = None
        self.jury = self.select_random_jury(num_members=3)
        self.image_size = (self.embedding_model.input_shape[1], self.embedding_model.input_shape[2])

    def select_random_jury(self, num_members):
        # take num_members jury members per class
        # s_df = sample_df(self.dataframe, self.config.df_config['cls_col'], n_sample_per_class=num_members,
        #                replace=False, random_state=111)
        s_df = sample_df(self.dataframe, self.config.df_config['cls_col'], n_sample_per_class=num_members, replace=False)
        return s_df[self.config.df_config['name_col']].values

    def predict_pairs(self, image, jury):
        preds = [
            self.model.predict(
                [tf.expand_dims(image, axis=0), tf.expand_dims(jury[i, :, :], axis=0)]
            ).flatten()[0]
            for i in range(len(jury))
        ]

        preds_sorted = sorted(zip(preds, self.jury), reverse=True)
        return preds_sorted

    # take an image and n other images and compare image against the jury
    def predict_image_by_jury(self, image_name):
        jury = self.get_images(self.jury)
        image = self.get_images([image_name])
        return self.predict_pairs(image, jury)

    def predict_dcim_image_by_jury(self, image_path):
        jury = self.get_images(self.jury)
        image = self.image_loader.load_dcim_image(image_path)
        return self.predict_pairs([image], jury)

    def get_images(self, image_names):
        return self.image_loader.load_image(self.config.images_dir, image_names, self.image_size)

    def plot_predictions(self, image_name, predictions):
        n = len(predictions) + 1
        columns = 5
        rows = round(n / columns)

        fig = plt.figure(figsize=(10, 10))
        ax = []

        print(f'Original {image_name}')
        print('Predictions (similarity, image name):')
        p = [print(pred) for pred in predictions]

        # plt.imshow(image[0])
        i = 1
        ax.append(fig.add_subplot(rows, columns, i))
        ax[-1].set_title('Original')
        ax[-1].set_xlabel(f'{image_name.split("_")[0]}')
        ax[-1].tick_params(labelsize=8)
        # ax[-1].yaxis.set_visible(False)
        ax[-1].axes.xaxis.set_ticks([])
        ax[-1].axes.yaxis.set_ticks([])

        image = self.get_images([image_name])

        plt.imshow(image[0])
        for pred, name in predictions:
            img = self.get_images([name])
            ax.append(fig.add_subplot(rows, columns, i + 1))
            ax[-1].axes.xaxis.set_ticks([])
            ax[-1].axes.yaxis.set_ticks([])
            label = name.split('_')[0]
            ax[-1].set_title(f'Similarity: {"{:.4f}".format(pred)}')
            ax[-1].set_xlabel(f'{label}')
            ax[-1].tick_params(bottom=False, top=False, labelsize=8)
            plt.imshow(img[0])
            i = i + 1

        plt.savefig(f'Prediction_{name}.png')
        plt.show()

    def show_prediction_dcim(self, image_path):
        jury = self.get_images(self.jury)
        image = self.image_loader.load_dcim_image(image_path)
        predictions = self.predict_pairs(image, jury)
        image_name = ntpath.basename(image.path)
        self.plot_predictions(image_name, predictions[image_name])

    def show_prediction(self, image_name):
        predictions = self.predict_images([image_name])
        self.plot_predictions(image_name, predictions[image_name])

    def predict_images(self, image_names):
        if isinstance(image_names, str):
            image_names = [image_names]
        predictions = {}
        jury = self.get_images(self.jury)
        images = self.get_images(image_names)
        images_processed = np.array([self.image_loader.preprocess_image(image) for image in images])

        for i, image in enumerate(image_names):
            prediction = self.predict_pairs(images_processed[i], jury)
            predictions[image] = prediction
        return predictions

    def dump_predictions(self, file_name):
        with open(file_name, 'w') as file:
            json_string = json.dumps(str(all_predictions))
            file.write(json_string)


class ImageLoader:
    def __init__(self, preprocess_input):
        self.preprocess_input = preprocess_input
        self.image_cache = {}

    def log(self, message):
        print(message)

    def cache_images(self, image_path, image_size):
        if os.path.exists(image_path):
            if image_path not in self.image_cache:
                image = cv2.imread(image_path)
                image = cv2.resize(image, image_size)
                tf.cast(image, tf.float32)
                # image = tf.image.resize_with_pad(image, image_size[0], image_size[1])
                self.image_cache[image_path] = image
            return self.image_cache[image_path]
        else:
            self.log(f'ImageLoader.cache_images: Path not found: {image_path}')
            return None

    def load_image(self, images_dir, image_names, image_size):
        if isinstance(image_names, str):
            image_names = [image_names]
        images = []
        for name in image_names:
            image_path = os.path.join(images_dir, name)
            image = self.cache_images(image_path, image_size)
            images.append(image)
        return np.array(images)

    def preprocess_image(self, image):
        return self.preprocess_input(image)

    def load_dcim_image(self, dcim_image_path):
        fpath = os.path.join(config.data_dir, 'temp.png')
        helper.dcm2png(dcim_image_path, fpath)
        image = self.cache_images(fpath)
        return image  # self.preprocess_image(image)


if __name__ == "__main__":
    # tdf = load_json_as_df(config.data_dir, 'mri-images')
    json_path = os.path.join(config.src_dir, 'mri-images.json')
    print(json_path)
    tdf = pd.read_json(json_path, orient='index')

    # col = 'perspective'
    col = config.df_config['cls_col']
    name_col = config.df_config['name_col']
    # col = 'perspective-sequence'

    # models for perspective
    siam_model = keras.models.load_model(config.siam_model_perspective_path)
    embedding_model = keras.models.load_model(config.embedding_model_perspective_path)

    siam_predictor = SiamesePredictor(config, tdf, siam_model, embedding_model)
    test_image_names = tdf[name_col].values
    test_image_classes = tdf[col]

    siam_predictor.show_prediction(test_image_names[len(test_image_names)-2])

    # all_predictions = siam_predictor.predict_images(test_image_names)

    sys.exit()
