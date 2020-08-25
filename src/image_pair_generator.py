import numpy as np
import cv2
import tensorflow as tf
import os

default_imagegen_parameters = dict(
    rescale=1. / 255,
    rotation_range=10.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # shear_range=0.0005,
    zoom_range=[0.9, 1.1],
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest',
)


# #### Create Image Generator
# See http://sujitpal.blogspot.com/2017/02/using-keras-imagedatagenerator-with.html
#
class ImagePairGenerator:

    def __init__(self, imagegen_parameters=None):
        self.imagegen_parameters = imagegen_parameters
        if self.imagegen_parameters is None:
            self.imagegen_parameters = default_imagegen_parameters
        self.image_cache = {}

    def cache_images(self, image_path, image_size):
        if image_path not in self.image_cache:
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)
            # image = tf.image.resize_with_pad(image, image_size[0], image_size[1])

            self.image_cache[image_path] = image
        return self.image_cache[image_path]

    def preprocess_images(self, datagen, image_names, image_size, images_dir, seed):
        """
        * Reads images given by image_names list from the images_dir directory.
        * Preprocesses images read by the transformations given with the datagen transformation configuration
        * Resizes images to the given image_size tuple
        * Rescales image data to [0, 1]
        """
        np.random.seed(seed)

        # initialize ndarray holding the transformed images
        # it has shape defined by: number of image_names, image width, image height, 3 colors
        X = np.zeros(((len(image_names),) + image_size + (3,)))

        for i, image_name in enumerate(image_names):
            image = self.cache_images(os.path.join(images_dir, image_name), image_size)
            image = datagen.random_transform(image)

            # rescale image (as rescale parameter has no effect)
            image = image / 255
            X[i] = image

        return np.array(X)

    def generate_image_pairs(
            self,
            image_pairs,
            image_pair_labels,
            batch_size,
            image_size,
            images_dir
    ):
        """
        Arguments:
            image_pairs, # list of pairs of image names
            image_pair_labels, # labels corresponding to the list of image pairs, 
                1 if image pair come from the same class, 0 if pair come from different classes
            batch_size, # batch size
            image_size, # the image size tuple to transform the imgages to
            images_dir # directory of the images

        """

        datagen_left = tf.keras.preprocessing.image.ImageDataGenerator(**self.imagegen_parameters)
        datagen_right = tf.keras.preprocessing.image.ImageDataGenerator(**self.imagegen_parameters)

        # True as long as no StopIteration exception is thrown in next()
        while True:

            num_recs = len(image_pairs)

            # permutation indices, used to permute images and class labels in sync
            indices = np.random.permutation(np.arange(num_recs))

            num_batches = num_recs // batch_size

            for idx in range(num_batches):
                # indices for batches, used for images and class labels
                batch_indices = indices[idx * batch_size: (idx + 1) * batch_size]

                batch = [image_pairs[i] for i in batch_indices]
                labels = [image_pair_labels[i] for i in batch_indices]

                # same seed for both pair elements, used for random_transform
                seed = np.random.randint(low=0, high=1000, size=1)[0]

                # a batch element 'b' is an image_pair 
                # which is of form [image_name1, image_name2]
                xleft = self.preprocess_images(
                    datagen_left,
                    [b[0] for b in batch],
                    image_size,
                    images_dir,
                    seed
                )

                xright = self.preprocess_images(
                    datagen_right,
                    [b[1] for b in batch],
                    image_size,
                    images_dir,
                    seed
                )

                yield [xleft, xright], np.array(labels)
