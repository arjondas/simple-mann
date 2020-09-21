import numpy as np
import os
import random
import tensorflow as tf
from scipy import misc
import math
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)

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
	images_labels = [
		(i, os.path.join(path, image))
		for i, path in zip(labels, paths)
		for image in sampler(os.listdir(path))
	]
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

		data_folder = config.get("data_folder", "./data/omniglot_resized")
		self.img_size = config.get("img_size", (28, 28))

		self.dim_input = np.prod(self.img_size)
		self.dim_output = self.num_classes

		character_folders = [
			os.path.join(data_folder, family, character)
			for family in os.listdir(data_folder)
			if os.path.isdir(os.path.join(data_folder, family))
			for character in os.listdir(os.path.join(data_folder, family))
			if os.path.isdir(os.path.join(data_folder, family, character))
		]

		random.seed(1)
		random.shuffle(character_folders)
		num_val = 100
		num_train = 1100
		self.metatrain_character_folders = character_folders[:num_train]
		self.metaval_character_folders = character_folders[
			num_train : num_train + num_val
		]
		self.metatest_character_folders = character_folders[num_train + num_val :]

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
		elif batch_type == "val":
			folders = self.metaval_character_folders
		else:
			folders = self.metatest_character_folders
		
		all_image_batches = np.empty(
			(batch_size, self.num_samples_per_class, self.num_classes, 784)
		)
		all_label_batches = np.empty(
			(batch_size, self.num_samples_per_class, self.num_classes, self.num_classes)
		)

		for i in range(batch_size):
			sampled_folders = random.sample(folders, self.num_classes)
			temp_labels = list(range(self.num_classes))
			random.shuffle(temp_labels)
			labels_and_images = get_images(
				sampled_folders,
				temp_labels,
				self.num_samples_per_class,
				False
			)
			if self.num_samples_per_class > 1:
				labels_and_images1 = []
				labels_and_images2 = []
				for idx in range(len(labels_and_images)):
					if (idx+1) % self.num_samples_per_class == 0:
						labels_and_images2.append(labels_and_images[idx])
					else:
						labels_and_images1.append(labels_and_images[idx])
				random.shuffle(labels_and_images1)
				random.shuffle(labels_and_images2)
				labels_and_images = labels_and_images1 + labels_and_images2

			labels, images = [x for x in zip(*labels_and_images)]

			labels = np.array(labels)
			labels = np.reshape(labels, (self.num_samples_per_class, self.num_classes))
			labels = (np.arange(self.num_classes) == labels[..., None]).astype(float)
			
			images = [image_file_to_array(img, 784) for img in images]
			images = np.array(images)            
			images = np.reshape(images, (self.num_samples_per_class, self.num_classes, 784))

			all_image_batches[i] = images
			all_label_batches[i] = labels

		return all_image_batches, all_label_batches


if __name__ == "__main__":
	data_generator = DataGenerator(5, 3)
	i, l = data_generator.sample_batch("train", 3)
	i = i[0]
	l = l[0]
	fig, axarr = plt.subplots(i.shape[0], i.shape[1])
	fig.tight_layout()
	for one in range(i.shape[0]):
		for two in range(i.shape[1]):
			img = i[one, two]
			lbl = l[one, two]
			img = np.reshape(img, (28, 28))
			axarr[one, two].imshow(img)
			axarr[one, two].set_title(f'{lbl}')
	plt.show()
	print(i.shape)
	print(l.shape)