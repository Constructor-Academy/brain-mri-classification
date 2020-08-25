# import the necessary packages
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
import copy

class GradCAM:
	def __init__(self, model, layerName=None):
		# store the model, the class index used to measure the class
		# activation map, and the layer to be used when visualizing
		# the class activation map
		self.model = model
		self.layerName = layerName

		# if the layer name is None, attempt to automatically find
		# the target output layer
		if self.layerName is None:
			self.layerName = self.find_target_layer()

	def get_img_array(self, img_path, size):
		# `img` is a PIL image of size 224x224
		img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
		# `array` is a float32 Numpy array of shape (224, 224, 3)
		array = tf.keras.preprocessing.image.img_to_array(img)
		# We add a dimension to transform our array into a "batch"
		# of size (1, 224, 224, 3)
		array = np.expand_dims(array, axis=0)
		return array

	def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(self.model.layers):
			# check to see if the layer has a 4D output
			if len(layer.output_shape) == 4:
				return layer.name

		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

	def compute_heatmap(self, img_path, img_size, clf_layer_names, eps=1e-8):
		# get image array
		img_array = self.get_img_array(img_path, img_size)

		# construct our gradient model by supplying (1) the inputs
		# to our pre-trained model, (2) the output of the (presumably)
		# final 4D layer in the network, and (3) the output of the
		# softmax activations from the model
		last_conv_layer = self.model.get_layer(self.layerName)

		conv_model = Model(
			inputs=[self.model.inputs],
			outputs=[last_conv_layer.output]
		)

		# second part of model connected to conv_model output
		clf_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
		x = clf_input
		for layer_name in clf_layer_names:
			x = self.model.get_layer(layer_name)(x)

		clf_model = Model(
			inputs=[clf_input],
			outputs=[x]
		)

		# record operations for automatic differentiation
		with tf.GradientTape(persistent=True) as tape:
			# Compute activations of the last conv layer and make the tape watch it
			conv_output = conv_model(img_array)
			tape.watch(conv_output)

			# Compute class predictions
			preds = clf_model(conv_output)
			top_pred_index = tf.argmax(preds[0])
			top_class_channel = preds[:, top_pred_index]

		# This is the gradient of the top predicted class with regard to
		# the output feature map of the last conv layer
		grads = tape.gradient(top_class_channel, conv_output)

		# This is a vector where each entry is the mean intensity of the gradient
		# over a specific feature map channel
		pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

		# We multiply each channel in the feature map array
		# by "how important this channel is" with regard to the top predicted class
		conv_output = conv_output.numpy()[0]
		pooled_grads = pooled_grads.numpy()
		for i in range(pooled_grads.shape[-1]):
			conv_output[:, :, i] *= pooled_grads[i]

		# The channel-wise mean of the resulting feature map
		# is our heatmap of class activation
		heatmap = np.mean(conv_output, axis=-1)

		# For visualization purpose, we will also normalize the heatmap between 0 & 1
		# then scale by 255 and convert to uint8
		heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + eps)
		heatmap = np.uint8(heatmap * 255)

		# return the resulting heatmap to the calling function
		return heatmap

	def overlay_heatmap(self, heatmap, image, alpha=0.5,
		colormap=cv2.COLORMAP_VIRIDIS):
		# apply the supplied color map to the heatmap and then
		# overlay the heatmap on the input image
		heatmap = cv2.applyColorMap(heatmap, colormap)
		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

		# return a 2-tuple of the color mapped heatmap and the output,
		# overlaid image
		return (heatmap, output)