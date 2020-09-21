import numpy as np
import random
import tensorflow as tf
from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1, 'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 16, 'Number of N-way classification tasks per batch')

def loss_function(preds, labels):
	"""
	Computes loss
	Args:
		preds: [B, K+1, N, N] network output
		labels: [B, K+1, N, N] labels
	Returns:
		scalar loss
	"""
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds[:,-1], labels=labels[:,-1]))


class MANN(tf.keras.Model):
	def __init__(self, num_classes, samples_per_class):
		super(MANN, self).__init__()
		self.num_classes = num_classes
		self.samples_per_class = samples_per_class
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.LSTM(256, return_sequences=True, activation='relu'))
		self.model.add(tf.keras.layers.Dense(num_classes))

	def call(self, input_images, input_labels):
		"""
		Simple MANN
		Args:
			input_images: [B, K+1, N, 784] flattened images
			labels: [B, K+1, N, N] ground truth labels
		Returns:
			[B, K+1, N, N] predictions
		"""
		input_labels = tf.concat([input_labels[:,:-1], tf.zeros_like(tf.expand_dims(input_labels[:,-1], axis=1))], axis=1)
		input_images = tf.reshape(input_images, tf.convert_to_tensor([-1, self.samples_per_class * self.num_classes, 784], dtype=tf.dtypes.int32))
		input_labels = tf.reshape(input_labels, tf.convert_to_tensor([-1, self.samples_per_class * self.num_classes, self.num_classes], dtype=tf.dtypes.int32))
		input_images = tf.concat([input_images, input_labels], -1)
		out = self.model(input_images)
		out = tf.reshape(out, tf.convert_to_tensor([-1, self.samples_per_class, self.num_classes, self.num_classes], dtype=tf.dtypes.int32))
		return out

ims = tf.placeholder(tf.float32, shape=(None, FLAGS.num_samples + 1, FLAGS.num_classes, 784))
labels = tf.placeholder(tf.float32, shape=(None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))

data_generator = DataGenerator(FLAGS.num_classes, FLAGS.num_samples + 1)

o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
out = o(ims, labels)

loss = loss_function(out, labels)
optim = tf.train.AdamOptimizer(0.001)
optimizer_step = optim.minimize(loss)
iterations = []
test_accuracys = []
with tf.Session() as sess:
	sess.run(tf.local_variables_initializer())
	sess.run(tf.global_variables_initializer())

	for step in range(50000):
		i, l = data_generator.sample_batch('train', FLAGS.meta_batch_size)
		feed = {ims: i.astype(np.float32), labels: l.astype(np.float32)}
		_, ls = sess.run([optimizer_step, loss], feed)

		if step % 100 == 0:
			print("*" * 5 + "Iteration " + str(step) + "*" * 5)
			i, l = data_generator.sample_batch('test', 100)
			feed = {ims: i.astype(np.float32), labels: l.astype(np.float32)}
			pred, tls = sess.run([out, loss], feed)
			print("Train Loss:", ls, "Test Loss:", tls)
			pred = pred.reshape(-1, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes)
			pred = pred[:, -1, :, :].argmax(2)
			l = l[:, -1, :, :].argmax(2)
			accuracy = (1.0 * (pred == l)).mean()
			iterations.append(step)
			test_accuracys.append(accuracy)
			print("Test Accuracy", accuracy)

plt.plot(iterations, test_accuracys)
plt.title('Simple MANN')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.savefig('simple_mann.png')