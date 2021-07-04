import dnnlib.tflib as tflib
from training import dataset
import numpy as np

tfrecord_dir = '../../datasets/cars_v5_512'

tflib.init_tf({'gpu_options.allow_growth': True})
training_set = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle_mb=0)
tflib.init_uninitialized_vars()

batch_size = 10

interpolation_mag = np.random.uniform(size=[batch_size])

labels = training_set.get_random_labels_np(batch_size)
print('before')
rotation_offset = 108
rotations = labels[:, rotation_offset:rotation_offset + 8]
rotation_index = np.argmax(rotations, axis=1)
new_rotation_index = ((rotation_index + np.random.choice([-1, 1], size=[batch_size])) % 8)
new_rotation = np.zeros([batch_size, 8], dtype=np.uint32)
new_rotation[np.arange(batch_size), new_rotation_index] = 1
new_rotation = new_rotation * np.expand_dims(np.max(labels[:, rotation_offset:rotation_offset + 8], axis=1).astype(np.uint32), axis=1)
labels_interpolate = training_set.get_random_labels_np(batch_size)
labels_interpolate[:, rotation_offset:rotation_offset + 8] = new_rotation

interpolation_mag_label = np.expand_dims(interpolation_mag, axis=-1)
mixed_label = labels * interpolation_mag_label + labels_interpolate * (1 - interpolation_mag_label)

print('after')
print(np.round(labels[:, rotation_offset:rotation_offset + 8], 3))
print(np.round(labels_interpolate[:, rotation_offset:rotation_offset + 8], 3))
print(np.round(mixed_label[:, rotation_offset:rotation_offset + 8], 3))
