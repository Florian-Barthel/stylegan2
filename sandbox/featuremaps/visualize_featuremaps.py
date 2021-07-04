from training import misc
import numpy as np
import dnnlib.tflib as tflib
from matplotlib import pyplot as plt
from training import dataset


tfrecord_dir = '../../datasets/cars_v5_512'

tflib.init_tf({'gpu_options.allow_growth': True})
dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, buffer_mb=1, shuffle_mb=0)
tflib.init_uninitialized_vars()

x, labels = dset.get_minibatch_np(2)
isolated_label = np.zeros(labels.shape)
isolated_label[0, 0] = 1
background_offset = 1 + 67 + 12 + 18 + 10 + 8 + 5
color_offset = 1 + 67
isolated_label[0, background_offset:] = labels[0, background_offset:]
# isolated_label[0, color_offset:color_offset + 12] = labels[0, color_offset:color_offset + 12]
# baseline model
network_pkl = '../../results/00117-stylegan2-cars_v5_512-2gpu-config-f/network-snapshot-000361.pkl'

_G, D_original, _Gs = misc.load_pkl(network_pkl)
D = tflib.Network('D', func_name='training.networks_stylegan2_discriminator_mod_cutoff.D_stylegan2')
D.copy_vars_from(D_original)
f_maps = D.get_output_for(x, isolated_label)
f_maps = f_maps.eval()


# fig, axs = plt.subplots(5, 5)
#
# counter = 0
# for i in range(5):
#     for j in range(5):
#         axs[i][j].imshow(f_maps[0, counter, :, :])
#         axs[i][j].axis('off')
#         counter += 1

plt.imshow(np.mean(f_maps[0], axis=0))

plt.savefig('fmaps_background_avg.png', dpi=400)
plt.show()

plt.imshow(np.transpose(x[0], [1, 2, 0]))
plt.axis('off')
plt.savefig('fmaps_input.png', dpi=300)
plt.show()
plt.close('all')
