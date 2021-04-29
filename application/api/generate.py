import dnnlib.tflib as tflib
import tensorflow as tf
import training.misc as misc
import numpy as np
from PIL import Image
import io
import base64


class Generate:
    def __init__(self):
        tflib.init_tf({'gpu_options.allow_growth': True})
        self.session = tflib.create_session(None, force_as_default=True)
        self.latent_placeholder = tf.placeholder(tf.float32, shape=(None, 512))
        self.dlatent_placeholder = tf.placeholder(tf.float32, shape=(None, 16, 512))
        self.label_placeholder = tf.placeholder(tf.float32, shape=(None, 127))
        self.Gs = None
        self.mapping = None
        self.synthesis = None

    def load_network(self, network_path):
        _G, _D, self.Gs = misc.load_pkl('../../' + network_path)
        if 'mapping' in self.Gs.components:
            self.mapping = self.Gs.components.mapping.get_output_for(self.latent_placeholder, self.label_placeholder)
        else:
            dlatent = self.Gs.components.mapping_latent.get_output_for(self.latent_placeholder)
            dlabel = self.Gs.components.mapping_label.get_output_for(self.label_placeholder)
            self.mapping = dlatent + dlabel
        self.synthesis = self.Gs.components.synthesis.get_output_for(self.dlatent_placeholder, randomize_noise=False)

    def generate_single_image(self, label, seed, size):
        dlatent = self._run_mapping(seed=seed, label=label)
        output = self.session.run(self.synthesis, feed_dict={self.dlatent_placeholder: dlatent})
        img = self.process_generator_output(output, size)[0]
        return self.convert_to_base64_str(img)

    def _generate_images(self, dlatents, size, batch_size=5, cache_size=25):
        result_images = []
        images_left = cache_size
        image_index = 0
        while images_left > 0:
            current_batch_size = min(images_left, batch_size)
            dlatent_interpolate = dlatents[image_index:image_index + current_batch_size]
            output = self.session.run(self.synthesis, feed_dict={self.dlatent_placeholder: dlatent_interpolate})
            result_images += self.process_generator_output(output, size)
            images_left -= current_batch_size
            image_index += current_batch_size
        return result_images

    @staticmethod
    def process_generator_output(output, size):
        img_list = []
        for i in range(np.shape(output)[0]):
            img = np.transpose(output[i], [1, 2, 0])
            img = img * 127.5 + 127.5
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
            img = img.resize((size, size))
            img_list.append(img)
        return img_list

    @staticmethod
    def convert_to_base64_str(img):
        raw_bytes = io.BytesIO()
        img.save(raw_bytes, "JPEG")
        raw_bytes.seek(0)
        return str(base64.b64encode(raw_bytes.read()))

    def _run_mapping(self, seed, label):
        latent = np.random.RandomState(seed).normal(0, 1, [1, self.Gs.input_shape[1]]).astype(np.float32)
        return self.session.run(self.mapping, feed_dict={self.latent_placeholder: latent,
                                                         self.label_placeholder: label})

    def interpolations(self, label_left, seed_left, label_right, seed_right, size=512, cache_size=25):
        dlatent_left = self._run_mapping(seed=seed_left, label=label_left)
        dlatent_right = self._run_mapping(seed=seed_right, label=label_right)
        interpolation_dlatents = []
        for i in range(cache_size):
            magnitude = i / (cache_size - 1)
            interpolation_dlatents.append(dlatent_left * (1 - magnitude) + dlatent_right * magnitude)
        interpolation_dlatents = np.concatenate(interpolation_dlatents, axis=0)
        interpolation_cache = self._generate_images(interpolation_dlatents, size=size, cache_size=cache_size)
        json_list = []
        for i in range(cache_size):
            json_list.append(self.convert_to_base64_str(interpolation_cache[i]))
        return json_list, cache_size

    def rotations(self, labels, seed, size=512, cache_size=20):
        json_list = []
        for i in range(8):
            interpolate_list, _ = self.interpolations(label_left=labels[i],
                                                      label_right=labels[(i + 1) % 8],
                                                      seed_left=seed, seed_right=seed,
                                                      size=size,
                                                      cache_size=cache_size)
            json_list += interpolate_list
        return json_list, cache_size * 8
