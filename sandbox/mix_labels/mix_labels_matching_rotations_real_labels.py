import dnnlib.tflib as tflib
from training import dataset
import numpy as np
import tensorflow as tf

tfrecord_dir = '../../datasets/cars_v5_512'

tflib.init_tf({'gpu_options.allow_growth': True})
training_set = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle_mb=0)
tflib.init_uninitialized_vars()

batch_size = 20
factor = 2

interpolation_mag = tf.random.uniform([batch_size], minval=0, maxval=1)

labels = training_set.get_random_labels_np(batch_size)
print('before')
rotation_offset = 108
print(labels[:, rotation_offset:rotation_offset + 8])
rotations = labels[:, rotation_offset:rotation_offset + 8]
rotation_index = np.argmax(rotations, axis=1)
shifted_rotation_index = ((rotation_index + np.random.choice([-1, 1], size=[batch_size])) % 8)

num_uniques = 0
labels_interpolate = training_set.get_random_labels_np(batch_size * factor)
while num_uniques < 8:
    labels_interpolate = np.concatenate([labels_interpolate, training_set.get_random_labels_np(batch_size * factor)], axis=0)
    interpolate_rotations = labels_interpolate[:, rotation_offset:rotation_offset + 8]
    interpolate_rotation_index = np.argmax(interpolate_rotations, axis=1)
    uniques, unique_indices = np.unique(interpolate_rotation_index, return_index=True)
    num_uniques = len(uniques)

labels_interpolate = labels_interpolate[unique_indices[shifted_rotation_index]]
labels_interpolate[:, 108:108 + 8] *= np.expand_dims(np.max(labels[:, rotation_offset:rotation_offset + 8], axis=1).astype(np.uint32), axis=1)
print(labels_interpolate[:, 108:108 + 8])

interpolation_mag_label = tf.expand_dims(interpolation_mag, axis=-1)
mixed_label = labels * interpolation_mag_label + labels_interpolate * (1 - interpolation_mag_label)

print('after')
print(np.round(mixed_label[:, rotation_offset:rotation_offset + 8].eval(), 3))
