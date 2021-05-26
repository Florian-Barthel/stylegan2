import dnnlib.tflib as tflib
import tensorflow as tf
import training.misc as misc
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt


class Generate:
    def __init__(self):
        self.session = None

        # Placeholder
        self.latent_placeholder = None
        self.dlatent_placeholder = None
        self.label_placeholder = None
        self.magnitude_placeholder = None
        self.dlatent_left_placeholder = None
        self.dlatent_right_placeholder = None

        # Network Paths
        self.Gs = None
        self.mapping = None
        self.synthesis = None
        self.dlatent_separate_mapping = None
        self.dlabel_separate_mapping = None
        self.gradients = None

        self.concat = None
        self.separate_mapping = None

    def load_network(self, network_path):
        self.session = tflib.create_session(None, force_as_default=True)
        _G, _D, self.Gs = misc.load_pkl('../../results/' + network_path)
        self.latent_placeholder = tf.placeholder(tf.float32, shape=(None, 512))
        self.label_placeholder = tf.placeholder(tf.float32, shape=(None, 127))

        self.dlatent_placeholder = tf.placeholder(tf.float32, shape=(None, 16, 512))
        self.dlatent_left_placeholder = tf.placeholder(tf.float32, shape=(None, 16, 512))
        self.dlatent_right_placeholder = tf.placeholder(tf.float32, shape=(None, 16, 512))

        if 'mapping' in self.Gs.components:
            self.separate_mapping = False
            self.mapping = self.Gs.components.mapping.get_output_for(self.latent_placeholder, self.label_placeholder)
        else:
            self.separate_mapping = True
            if self.Gs.components.synthesis.input_shape[2] > 512:
                self.concat = True
                self.dlatent_placeholder = tf.placeholder(tf.float32, shape=(None, 16, 1024))
                self.dlatent_left_placeholder = tf.placeholder(tf.float32, shape=(None, 16, 1024))
                self.dlatent_right_placeholder = tf.placeholder(tf.float32, shape=(None, 16, 1024))
            else:
                self.concat = False
            self.dlatent_separate_mapping = self.Gs.components.mapping_latent.get_output_for(self.latent_placeholder)
            self.dlabel_separate_mapping = self.Gs.components.mapping_label.get_output_for(self.label_placeholder)
        self.synthesis = self.Gs.components.synthesis.get_output_for(self.dlatent_placeholder, randomize_noise=False)

        self.magnitude_placeholder = tf.placeholder(tf.float32, shape=())
        dlatent_interpolate = self.dlatent_left_placeholder * (
                    1 - self.magnitude_placeholder) + self.dlatent_right_placeholder * self.magnitude_placeholder
        self.gradients = tf.gradients(
            tf.reduce_sum(
                self.Gs.components.synthesis.get_output_for(
                    dlatent_interpolate, randomize_noise=False
                )
            ), [self.dlatent_left_placeholder]
        )

    def generate_single_image(
            self,
            label,
            seed,
            size):
        dlatent = self._mapping(seed, label)
        output = self.session.run(self.synthesis, feed_dict={self.dlatent_placeholder: dlatent})
        img = self.process_generator_output(output, size)[0]
        return self.convert_to_base64_str(img)

    def interpolations(
            self,
            label_left,
            seed_left,
            label_right,
            seed_right,
            interpolate_in_z=False,
            size=512,
            cache_size=25):
        interpolation_dlatents = []
        dlatent_left = self._mapping(seed_left, label_left, only_latent=interpolate_in_z)
        dlatent_right = self._mapping(seed_right, label_right, only_latent=interpolate_in_z)
        for i in range(cache_size):
            magnitude = i / (cache_size - 1)
            dlatent_interpolate = dlatent_left * (1 - magnitude) + dlatent_right * magnitude
            if interpolate_in_z:
                label_interpolate = label_left * (1 - magnitude) + label_right * magnitude
                dlabel_interpolate = self.session.run(
                    self.dlabel_separate_mapping,
                    feed_dict={self.label_placeholder: label_interpolate}
                )
                interpolation_dlatents.append(self._combine_latent_and_label(dlatent_interpolate, dlabel_interpolate))
            else:
                interpolation_dlatents.append(dlatent_interpolate)
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
                                                      seed_left=seed,
                                                      seed_right=seed,
                                                      size=size,
                                                      cache_size=cache_size)
            json_list += interpolate_list
        return json_list, cache_size * 8

    def interpolation_graph(
            self,
            label_left,
            seed_left,
            label_right,
            seed_right,
            num_steps):
        plt.figure()
        dlatent_left = self._mapping(seed_left, label_left)
        dlatent_right = self._mapping(seed_right, label_right)

        x_axis = []
        y_axis = []
        for j in range(num_steps):
            magnitude = j / num_steps
            x_axis.append(magnitude)
            pl_grads = self.session.run(
                self.gradients,
                feed_dict={self.dlatent_left_placeholder: dlatent_left,
                           self.dlatent_right_placeholder: dlatent_right,
                           self.magnitude_placeholder: magnitude}
            )
            pl = pl_grads[0] / (1 - magnitude)
            pl_lengths = np.sqrt(np.mean(np.sum(np.square(pl), axis=2), axis=1))
            y_axis.append(pl_lengths[0])
        linear_diff = []
        linear_func = []
        for j in range(num_steps):
            x = j / num_steps
            f = y_axis[0] + x * (y_axis[-1] - y_axis[0]) / x_axis[-1]
            linear_func.append(f)
            linear_diff.append(np.abs(y_axis[j] - f) / np.abs(y_axis[0] - y_axis[-1]))

        linear_diff = np.mean(linear_diff)

        plt.plot(x_axis, y_axis)
        plt.plot(x_axis, linear_func)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_buf = Image.open(buf)
        graph_image = self.convert_to_base64_str(image_buf)
        buf.close()
        plt.clf()
        return graph_image, linear_diff

    def _combine_latent_and_label(self, dlatent, dlabel):
        if self.concat:
            return np.concatenate([dlatent, dlabel], axis=-1)
        else:
            return dlatent + dlabel

    def _mapping(self, seed, label, only_latent=False):
        latent = np.random.RandomState(seed).normal(0, 1, [1, self.Gs.input_shape[1]]).astype(np.float32)
        if self.separate_mapping:
            dlatent = self.session.run(self.dlatent_separate_mapping, feed_dict={self.latent_placeholder: latent})
            if only_latent:
                return dlatent
            dlabel = self.session.run(self.dlabel_separate_mapping, feed_dict={self.label_placeholder: label})
            dlatent = self._combine_latent_and_label(dlatent, dlabel)
        else:
            dlatent = self.session.run(self.mapping, feed_dict={self.latent_placeholder: latent,
                                                                self.label_placeholder: label})
        return dlatent

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
        if not img.mode == 'RGB':
            img = img.convert('RGB')
        raw_bytes = io.BytesIO()
        img.save(raw_bytes, "JPEG")
        raw_bytes.seek(0)
        return str(base64.b64encode(raw_bytes.read()))
