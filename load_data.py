import numpy as np
import os
import random
import tensorflow as tf
from scipy import misc
import math

def get_images(paths, labels, nb_samples=None, shuffle=True):
	"""
	Takes a set of character folders and labels and returns paths to image files
	paired with labels.
	Args:
		paths: A list of character folders
		labels: List or numpy array of same length as paths
		nb_samples: Number of images to retrieve per character
	Returns:
		List of (label, image_path) tuples
	"""
	if nb_samples is not None:
		sampler = lambda x: random.sample(x, nb_samples)
	else:
		sampler = lambda x: x
	images_labels = [(i, os.path.join(path, image))
					 for i, path in zip(labels, paths)
					 for image in sampler(os.listdir(path))]
	if shuffle:
		random.shuffle(images_labels)
	return images_labels



def image_file_to_array(filename, dim_input):
	"""
	Takes an image path and returns numpy array
	Args:
		filename: Image filename
		dim_input: Flattened shape of image
	Returns:
		1 channel image
	"""
	image = misc.imread(filename)
	image = image.reshape([dim_input])
	image = image.astype(np.float32) / 255.0
	image = 1.0 - image
	return image


class DataGenerator(object):
	"""
	Data Generator capable of generating batches of Omniglot data.
	A "class" is considered a class of omniglot digits.
	"""

	def __init__(self, num_classes, num_samples_per_class, config={}):
		"""
		Args:
			num_classes: Number of classes for classification (K-way)
			num_samples_per_class: num samples to generate per class in one batch
			batch_size: size of meta batch size (e.g. number of functions)
		"""
		self.num_samples_per_class = num_samples_per_class
		self.num_classes = num_classes

		data_folder = config.get('data_folder', './omniglot_resized')
		self.img_size = config.get('img_size', (28, 28))
		
		self.dim_input = np.prod(self.img_size)
		self.dim_output = self.num_classes
		# print('Given folder name',data_folder)
		character_folders = [os.path.join(data_folder, family, character)
							 for family in os.listdir(data_folder)
							 if os.path.isdir(os.path.join(data_folder, family))
							 for character in os.listdir(os.path.join(data_folder, family))
							 if os.path.isdir(os.path.join(data_folder, family, character))]

		random.seed(1)
		random.shuffle(character_folders)
		num_val = 100
		num_train = 1100
		self.index = 0
		self.val_index = 0
		self.test_index = 0
		self.metatrain_character_folders = character_folders[: num_train]
		self.metaval_character_folders = character_folders[
			num_train:num_train + num_val]
		self.metatest_character_folders = character_folders[
			num_train + num_val:]
	def sample_batch(self, batch_type, batch_size):
		"""
		Samples a batch for training, validation, or testing
		Args:
			batch_type: train/val/test
		Returns:
			A a tuple of (1) Image batch and (2) Label batch where
			image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
			where B is batch size, K is number of samples per class, N is number of classes
		"""
		if batch_type == "train":
			folders = self.metatrain_character_folders
			# folders = folders[
       		# 	(self.index % math.ceil(len(self.metatrain_character_folders) / (batch_size * self.num_classes))):]
			# self.index = self.index + 1
		elif batch_type == "val":
			folders = self.metaval_character_folders
			# folders = folders[
       		# 	(self.val_index % math.ceil(len(self.metatrain_character_folders) / (batch_size * self.num_classes))):]
			# self.val_index = self.val_index + 1
		else:
			folders = self.metatest_character_folders
			# folders = folders[
       		# 	(self.test_index % math.ceil(len(self.metatrain_character_folders) / (batch_size * self.num_classes))):]
			# self.test_index = self.test_index + 1

		#############################
		#### YOUR CODE GOES HERE ####
		all_image_batches = np.empty((batch_size, self.num_samples_per_class, self.num_classes, 784))
		all_label_batches = np.empty((batch_size, self.num_samples_per_class, self.num_classes, self.num_classes))
		random.shuffle(folders)
		rem_classes = batch_size * self.num_classes - len(folders)
		folders = folders + (random.sample(folders, rem_classes) if rem_classes > 0 else [])
		for i in range(batch_size):
			n_folders = folders[i*self.num_classes: (i+1)*self.num_classes]
			labels_and_images = get_images(n_folders, [i+1 for i in range(self.num_classes)], self.num_samples_per_class, True)
			# print('len of l&i',len(labels_and_images))
			# print('******************')
			labels, images = [x for x in zip(*labels_and_images)]
			labels = np.array(labels)
			labels = np.reshape(labels, (self.num_classes, self.num_samples_per_class))
			labels = np.swapaxes(labels, 0, 1)
			labels = (np.arange(self.num_classes) == labels[...,None]).astype(float)

			images = [image_file_to_array(i, 784) for i in images]
			images = np.array(images)
			images = np.reshape(images, (self.num_classes, self.num_samples_per_class, 784))
			images = np.swapaxes(images, 0, 1)
			all_image_batches[i,] = images
			all_label_batches[i,] = labels
		#############################

		return all_image_batches, all_label_batches


if __name__ == '__main__':
    data_generator = DataGenerator(5, 1 + 1)
    i, l = data_generator.sample_batch('test', 5)
    print(i.shape)
    print(l.shape)
    hh = np.concatenate((i,l), -1)
    print(hh.shape)