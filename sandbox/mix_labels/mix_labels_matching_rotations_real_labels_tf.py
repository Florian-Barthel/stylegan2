import dnnlib.tflib as tflib
from training import dataset
import numpy as np
import tensorflow as tf

tfrecord_dir = '../../datasets/cars_v5_512'

tflib.init_tf({'gpu_options.allow_growth': True})
training_set = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle_mb=0)
tflib.init_uninitialized_vars()

def label_vector(model=0, color=0, manufacturer=0, body=0, rotation=0, ratio=0, background=0):
    onehot = np.zeros((1, 127), dtype=np.float32)
    onehot[0, 0] = 1
    if model >= 0:
        onehot[0, 1 + model] = 1.0
    if color >= 0:
        onehot[0, 1 + 67 + color] = 1.0
    if manufacturer >= 0:
        onehot[0, 1 + 67 + 12 + manufacturer] = 1.0
    if body >= 0:
        onehot[0, 1 + 67 + 12 + 18 + body] = 1.0
    if rotation >= 0:
        onehot[0, 1 + 67 + 12 + 18 + 10 + rotation] = 1.0
    onehot[0, 1 + 67 + 12 + 18 + 10 + 8 + ratio] = 1.0
    if background >= 0:
        onehot[0, 1 + 67 + 12 + 18 + 10 + 8 + 5 + background] = 1.0
    return np.array(onehot)

label = label_vector(rotation=0)
for i in range(1, 8):
    label = np.concatenate([label, label_vector(rotation=i)])
label = np.array(label)
labels = tf.constant(label)

minibatch_size = tf.constant(8)
factor = 2

# Interpolate all labels by interpolation_mag
# labels = training_set.get_random_labels_tf(minibatch_size)
rotation_offset = 108
num_rotations = tf.constant(8)
rotations = labels[:, rotation_offset:rotation_offset + num_rotations]
rotation_index = tf.cast(tf.argmax(rotations, axis=1), dtype=tf.int32)
shift_directions = tf.constant([-1, 1], dtype=tf.int32)
shift = tf.gather(shift_directions, indices=tf.cast(tf.math.round(tf.random.uniform([minibatch_size], 0, 1)), dtype=tf.int32))
shifted_rotation_index = tf.math.mod(rotation_index + shift, tf.constant(8))
print(rotation_index.eval())
print(shifted_rotation_index.eval())

num_uniques = 0
labels_interpolate = training_set.get_random_labels_tf(minibatch_size * factor)
while num_uniques < num_rotations:
    labels_interpolate = tf.concat(
        [labels_interpolate, training_set.get_random_labels_tf(minibatch_size * factor)], axis=0)
    interpolate_rotations = labels_interpolate[:, rotation_offset:rotation_offset + num_rotations]
    interpolate_rotation_index = np.argmax(interpolate_rotations, axis=1)
    uniques, unique_indices = np.unique(interpolate_rotation_index, return_index=True)
    num_uniques = tf.shape(uniques)[0]

labels_interpolate = labels_interpolate[unique_indices[shifted_rotation_index]]
labels_interpolate[:, rotation_offset:rotation_offset + num_rotations] *= np.expand_dims(
    np.max(rotations, axis=1).astype(np.uint32), axis=1)

dlabel_left = G.components.mapping_label.get_output_for(labels, is_training=True)
dlabel_right = G.components.mapping_label.get_output_for(labels_interpolate, is_training=True)
dlabel = dlabel_left * interpolation_mag_dlatent + dlabel_right * (1 - interpolation_mag_dlatent)

print('after')
print(np.round(mixed_label[:, rotation_offset:rotation_offset + 8].eval(), 3))
