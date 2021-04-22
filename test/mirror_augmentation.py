import dnnlib.tflib as tflib
from training import dataset
import numpy as np

tfrecord_dir = '../datasets/cars_v4_512'

tflib.init_tf({'gpu_options.allow_growth': True})
dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle_mb=0)
tflib.init_uninitialized_vars()

x, labels = dset.get_minibatch_np(20)
print('before')
print(labels)
random_vector = np.random.uniform(0, 1, [np.shape(x)[0]]) < 0.5
# x = np.where(random_vector, x, np.flip(x, axis=3))
rotation_offset = 1 + 67 + 12 + 18 + 10

indices_first = np.arange(rotation_offset)
swaps = np.array([0, 2, 1, 4, 3, 5, 7, 6]) + rotation_offset
indices_last = np.arange(rotation_offset + 8, np.shape(labels)[1])
indices = np.concatenate([indices_first, swaps, indices_last], axis=0)
# indices = np.broadcast_to(indices, [np.shape(labels)[0], np.shape(labels)[1]])
mirrored_labels = np.take(labels, indices, axis=1) # same as gather
labels = np.where(random_vector, labels, mirrored_labels)
print('after')
print(labels)