import dnnlib.tflib as tflib
from training import dataset
import numpy as np
import tensorflow as tf

tfrecord_dir = '../datasets/cars_v4_512'

tflib.init_tf({'gpu_options.allow_growth': True})
dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle_mb=0)
tflib.init_uninitialized_vars()

x, labels = dset.get_minibatch_np(4)
print('before')
print(labels)
interpolation_mag = tf.random.normal([1], 0.5, 0.15)

# mixed_label = tf.expand_dims(tf.where(labels[0] != labels[1], tf.fill([tf.shape(labels)[1]], 0.5), labels[0]), axis=0)

wx1 = tf.where(labels[0] > labels[1], labels[0] * interpolation_mag, labels[0])
wx2 = tf.where(labels[0] < labels[1], labels[1] * (1 - interpolation_mag), labels[1])
mixed_label = tf.expand_dims(tf.clip_by_value(wx1 + wx2, 0, 1), axis=0)

# 6. replace labels[0] with the new mixed label
labels = tf.concat([mixed_label, labels[1:]], axis=0)
print('after')
print(labels.eval())