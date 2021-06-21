# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Frechet Inception Distance (FID)."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib
from tqdm import tqdm

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

class FID(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu

    @staticmethod
    def _combine_latent_and_label(dlatent, dlabel, Gs_clone):
        if Gs_clone.components.synthesis.input_shape[2] > 512:
            return tf.concat([dlatent, dlabel], axis=-1)
        else:
            return dlatent + dlabel

    def _interpolate_random(self, Gs_clone, Gs_kwargs):
        magnitude = tf.random_uniform(shape=[self.minibatch_per_gpu], minval=0, maxval=1, dtype=tf.float32)

        latents = np.random.normal(size=[self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
        labels_right = self._get_random_labels_np(self.minibatch_per_gpu)
        rotation_offset = 108
        rotations = labels_right[:, rotation_offset:rotation_offset + 8]
        rotation_index = np.argmax(rotations, axis=1)
        shifted_rotation_index = ((rotation_index + np.random.choice([-1, 1], size=[self.minibatch_per_gpu])) % 8)
        new_rotation = np.zeros([self.minibatch_per_gpu, 8], dtype=np.uint32)
        new_rotation[np.arange(self.minibatch_per_gpu), shifted_rotation_index] = 1
        new_rotation = new_rotation * np.expand_dims(
            np.max(labels_right[:, rotation_offset:rotation_offset + 8], axis=1).astype(np.uint32), axis=1)
        labels_left = labels_right.copy()
        labels_left[:, rotation_offset:rotation_offset + 8] = new_rotation

        if 'mapping' in Gs_clone.components:
            dlatent_left = Gs_clone.components.mapping.get_output_for(latents, labels_left)
            dlatent_right = Gs_clone.components.mapping.get_output_for(latents, labels_right)
        else:
            dlatents = Gs_clone.components.mapping_latent.get_output_for(latents)
            dlabels_left = Gs_clone.components.mapping_label.get_output_for(labels_left)
            dlatent_left = self._combine_latent_and_label(dlatents, dlabels_left, Gs_clone)
            dlabels_right = Gs_clone.components.mapping_label.get_output_for(labels_right)
            dlatent_right = self._combine_latent_and_label(dlatents, dlabels_right, Gs_clone)
        magnitude = tf.expand_dims(tf.expand_dims(magnitude, axis=-1), axis=-1)
        dlatents_interpolate = dlatent_left * (1 - magnitude) + dlatent_right * magnitude
        return Gs_clone.components.synthesis.get_output_for(dlatents_interpolate, **Gs_kwargs)

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        inception = misc.load_pkl('https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/inception_v3_features.pkl')
        activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)

        # Calculate statistics for reals.
        cache_file = self._get_cache_file_for_reals(num_images=self.num_images)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if os.path.isfile(cache_file):
            mu_real, sigma_real = misc.load_pkl(cache_file)
        else:
            for idx, images in enumerate(self._iterate_reals(minibatch_size=minibatch_size)):
                begin = idx * minibatch_size
                end = min(begin + minibatch_size, self.num_images)
                activations[begin:end] = inception.run(images[:end-begin], num_gpus=num_gpus, assume_frozen=True)
                if end == self.num_images:
                    break
            mu_real = np.mean(activations, axis=0)
            sigma_real = np.cov(activations, rowvar=False)
            misc.save_pkl((mu_real, sigma_real), cache_file)

        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                inception_clone = inception.clone()
                images = self._interpolate_random(Gs_clone, Gs_kwargs)
                images = tflib.convert_images_to_uint8(images)
                result_expr.append(inception_clone.get_output_for(images))

        # Calculate statistics for fakes.
        for begin in tqdm(range(0, self.num_images, minibatch_size)):
            self._report_progress(begin, self.num_images)
            end = min(begin + minibatch_size, self.num_images)
            activations[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end-begin]
        mu_fake = np.mean(activations, axis=0)
        sigma_fake = np.cov(activations, rowvar=False)

        # Calculate FID.
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        self._report_result(np.real(dist))

#----------------------------------------------------------------------------
