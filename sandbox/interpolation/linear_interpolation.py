import dnnlib.tflib as tflib
import training.misc as misc
import tensorflow as tf
import numpy as np
import training.dataset as dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL.Image
import os

session = tflib.create_session(None, force_as_default=True)
latent_placeholder = tf.placeholder(tf.float32, shape=(None, 512))
dlatent_placeholder = tf.placeholder(tf.float32, shape=(None, 16, 512))
label_placeholder = tf.placeholder(tf.float32, shape=(None, 127))
_G, _D, Gs = misc.load_pkl('../results/00037-stylegan2-cars_v4_512-2gpu-config-f/network-snapshot-006316.pkl')

mapping = Gs.components.mapping.get_output_for(latent_placeholder, label_placeholder)

synthesis = Gs.components.synthesis.get_output_for(dlatent_placeholder)


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


for i in range(1):
    fig, ax = plt.subplots()
    rotation_order = [0, 1, 3, 6, 5, 7, 4, 2]
    rotation = 2  # np.random.randint(8)
    model = 1  # np.random.randint(67)
    color = 0  # np.random.randint(12)
    body = 0  # np.random.randint(10)
    manufacturer = 1  # np.random.randint(18)
    ratio = 1  # np.random.randint(5)
    background = 2  # np.random.choice([0, 5])
    seed = 1 # np.random.randint(10000)

    folder = 'interpolations_with_seed'
    os.makedirs(folder, exist_ok=True)
    description = 'ma{}-mo{}-co{}-bo{}-ro{}-ba{}-ra{}-se{}'.format(manufacturer, model, color, body, rotation,
                                                                   background, ratio, seed)

    label1 = label_vector(model,
                          color,
                          manufacturer,
                          body,
                          rotation_order[rotation],
                          ratio,
                          background)

    latent = np.random.RandomState(seed).normal(0, 1, [1, Gs.input_shape[1]]).astype(
        np.float32)  # np.random.normal(0, 1, [1, Gs.input_shape[1]]).astype(np.float32)

    fake_dlatents_out1 = Gs.components.mapping.get_output_for(latent, label1)
    fake_images_out1 = Gs.components.synthesis.get_output_for(fake_dlatents_out1)

    label2 = label_vector(model,
                          color,
                          manufacturer,
                          body,
                          rotation_order[(rotation + 1) % 8],
                          ratio,
                          background)
    # latent2 = np.random.normal(0, 1, [1, Gs.input_shape[1]]).astype(np.float32)

    fake_dlatents_out2 = Gs.components.mapping.get_output_for(latent, label2)
    fake_images_out2 = Gs.components.synthesis.get_output_for(fake_dlatents_out2)

    y = []
    w = 512
    h = 512
    interpolation_steps = 15
    canvas = PIL.Image.new('RGB', (w * interpolation_steps, h), 'white')
    for j in tqdm(range(interpolation_steps)):
        magnitude = j / interpolation_steps
        dlatent_intepolate = fake_dlatents_out1 * (1 - magnitude) + fake_dlatents_out2 * magnitude
        fake_images_out = Gs.components.synthesis.get_output_for(dlatent_intepolate)

        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out), [fake_dlatents_out1])[0] / (1 - magnitude)
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
        y.append(pl_lengths.eval()[0])
        image = tflib.convert_images_to_uint8(fake_images_out, nchw_to_nhwc=True).eval()
        canvas.paste(PIL.Image.fromarray(image[0], 'RGB'), (j * w, 0))
    canvas.save(os.path.join(folder, 'single-{}-image-{}.png'.format(i, description)))
    start = y[0]
    end = y[-1]

    linear_diff = []
    prev_diff = []
    linear_func = []
    for j in range(interpolation_steps):
        x = j / interpolation_steps
        f = start + x * (end - start) / ((interpolation_steps - 1) / interpolation_steps)
        linear_func.append(f)
        linear_diff.append(np.abs(y[j] - f) / np.abs(max(y) - min(y)))
        if j > 0:
            prev_diff.append(np.abs(y[j] - y[j - 1]) / np.abs(max(y) - min(y)))

    ax.plot(y)
    print('linear difference:', np.mean(linear_diff))
    print('previous difference:', np.mean(prev_diff))
    ax.plot(linear_func)

    fig.savefig(os.path.join(folder, 'single-{}-graph-{}.png'.format(i, description)), dpi=400)
    plt.show()
