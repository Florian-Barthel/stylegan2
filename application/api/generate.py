import dnnlib.tflib as tflib
import training.misc as misc
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from flask import jsonify

last_interpolation_request = [None, None, None, None]
last_interpolation_answer = None

session = tflib.create_session(None, force_as_default=True)
latent_placeholder = tf.placeholder(tf.float32, shape=(None, 512))
dlatent_placeholder = tf.placeholder(tf.float32, shape=(None, 16, 512))
label_placeholder = tf.placeholder(tf.float32, shape=(None, 127))
_G, _D, Gs = misc.load_pkl('../../results/00037-stylegan2-cars_v4_512-2gpu-config-f/network-snapshot-006557.pkl')
gen_image = Gs.get_output_for(latent_placeholder, label_placeholder, is_validation=True, randomize_noise=False,
                              truncation_psi_val=1.0)
mapping = Gs.components.mapping.get_output_for(latent_placeholder, label_placeholder, is_validation=True,
                                               truncation_psi_val=1.0)
synthesis = Gs.components.synthesis.get_output_for(dlatent_placeholder, is_validation=True, randomize_noise=False,
                                                   truncation_psi_val=1.0)


def label_vector(model, color, manufacturer, body, rotation, ratio, background):
    onehot = np.zeros((1, 127), dtype=np.float)
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


def single_image(label, seed, size):
    latent = np.random.RandomState(seed).normal(0, 1, [1, Gs.input_shape[1]]).astype(np.float32)
    image = session.run(gen_image, feed_dict={latent_placeholder: latent, label_placeholder: label})[0]
    image = np.transpose(image, [1, 2, 0])
    image = image * 127.5 + 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    img = Image.fromarray(image)
    img = img.resize((size, size))
    return img


def interpolations(label_left, seed_left, label_right, seed_right, batch_size=16, cache_size=32):
    global last_interpolation_answer
    global last_interpolation_request

    if np.array_equal([seed_left, seed_right],last_interpolation_request[:2]):
        if np.array_equal([label_left, label_right], last_interpolation_request[2:]):
            return last_interpolation_answer
    last_interpolation_request = [seed_left, seed_right, label_left, label_right]
    interpolation_cache = np.zeros([cache_size, 512, 512, 3], dtype=np.uint8)
    latent_left = np.random.RandomState(seed_left).normal(0, 1, [1, Gs.input_shape[1]]).astype(np.float32)
    latent_right = np.random.RandomState(seed_right).normal(0, 1, [1, Gs.input_shape[1]]).astype(np.float32)
    dlatent_left = session.run(mapping, feed_dict={latent_placeholder: latent_left, label_placeholder: label_left})[0]
    dlatent_right = session.run(mapping, feed_dict={latent_placeholder: latent_right, label_placeholder: label_right})[0]

    interpolation_dlatents = []
    for i in range(cache_size):
        magnitude = i / (cache_size - 1)
        interpolation_dlatents.append(dlatent_left * (1 - magnitude) + dlatent_right * magnitude)

    images_left = cache_size
    image_index = 0

    while images_left > 0:
        current_batch_size = min(images_left, batch_size)
        dlatent_interpolate = interpolation_dlatents[image_index:image_index + current_batch_size]
        image = session.run(synthesis, feed_dict={dlatent_placeholder: dlatent_interpolate})
        image = np.transpose(image, [0, 2, 3, 1])
        image = image * 127.5 + 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        interpolation_cache[image_index:image_index + current_batch_size] = image
        images_left -= current_batch_size
        image_index += current_batch_size

    json_list = []
    for i in range(cache_size):
        img = Image.fromarray(interpolation_cache[i])
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        json_list.append(str(img_base64))
    last_interpolation_answer = jsonify({'status': json_list, 'cache_size': cache_size})
    return last_interpolation_answer
