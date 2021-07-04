import dnnlib.tflib as tflib
import training.misc as misc
import tensorflow as tf
import numpy as np
import training.dataset as dataset
import matplotlib.pyplot as plt


session = tflib.create_session(None, force_as_default=True)
latent_placeholder = tf.placeholder(tf.float32, shape=(None, 512))
dlatent_placeholder = tf.placeholder(tf.float32, shape=(None, 16, 512))
label_placeholder = tf.placeholder(tf.float32, shape=(None, 127))
_G, _D, Gs = misc.load_pkl('../results/00070-stylegan2-cars_v4_512-2gpu-config-f/network-snapshot-007219.pkl')
#_G, _D, Gs = misc.load_pkl('../results/00087-stylegan2-cars_v4_512-4gpu-config-f/network-snapshot-006256.pkl')

mapping_latent = Gs.components.mapping_latent.get_output_for(latent_placeholder, is_validation=True, truncation_psi_val=1.0)
mapping_label = Gs.components.mapping_label.get_output_for(label_placeholder, is_validation=True, truncation_psi_val=1.0)

synthesis = Gs.components.synthesis.get_output_for(dlatent_placeholder, is_validation=True, randomize_noise=False, truncation_psi_val=1.0)

#mapping = Gs.components.mapping.get_output_for(latent_placeholder, label_placeholder, is_validation=True,
#                                               truncation_psi_val=1.0)

def label_vector(model, color, manufacturer, body, rotation, ratio, background):
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
    return onehot


# label = label_vector(0, 0, 0, 0, 0, 0, 0)
dlatent_lengths = []
dlabel_lengths = []
combine_lengths = []
num = 200
for i in range(num):
    dataset_obj = dataset.load_dataset(data_dir='../datasets', tfrecord_dir='cars_v4_512')
    label = dataset_obj.get_random_labels_np(1)
    latent = np.random.normal(0, 1, [1, Gs.input_shape[1]]).astype(np.float32)

    dlatent = session.run(mapping_latent, feed_dict={latent_placeholder: latent})[0]
    dlabel = session.run(mapping_label, feed_dict={label_placeholder: label})[0]

    #dlatent = session.run(mapping, feed_dict={latent_placeholder: latent, label_placeholder: label})[0]

    dlatent_length = np.linalg.norm(dlatent, 1)
    dlabel_length = np.linalg.norm(dlabel, 1)
    combine_length = np.linalg.norm(dlatent + dlabel, 1)
    #
    dlatent_lengths.append(dlatent_length)
    dlabel_lengths.append(dlabel_length)
    combine_lengths.append(combine_length)

print(np.mean(dlatent_lengths))
print(np.var(dlatent_lengths))

print(np.mean(dlabel_lengths))
print(np.var(dlabel_lengths))

print(np.mean(combine_lengths))
print(np.var(combine_lengths))
