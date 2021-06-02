from metrics import metric_base
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class IL(metric_base.MetricBase):
    def __init__(self, num_images, interpolation_steps, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.interpolation_steps = interpolation_steps

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        with tf.device('/gpu:%d' % 0):
            linear_diff = 0 # tf.constant(0, dtype=tf.float16)
            for _ in tqdm(range(self.num_images)):
                latent = np.random.normal(0, 1, [1, Gs.input_shape[1]]).astype(np.float32)
                labels = self._get_label_one_diff()
                if 'mapping' in Gs.components:
                    dlatent1 = Gs.components.mapping.get_output_for(latent, labels[0])
                    dlatent2 = Gs.components.mapping.get_output_for(latent, labels[1])
                else:
                    dlatent = Gs.components.mapping_latent.get_output_for(latent)
                    dlabel_1 = Gs.components.mapping_label.get_output_for(labels[0])
                    dlabel_2 = Gs.components.mapping_label.get_output_for(labels[1])
                    if Gs.components.synthesis.input_shape[2] > 512:
                        dlatent1 = tf.concat([dlatent, dlabel_1], axis=-1)
                        dlatent2 = tf.concat([dlatent, dlabel_2], axis=-1)
                    else:
                        dlatent1 = dlatent + dlabel_1
                        dlatent2 = dlatent + dlabel_2

                y = []
                for j in range(self.interpolation_steps):
                    magnitude = j / self.interpolation_steps
                    dlatent_intepolate = dlatent1 * (1 - magnitude) + dlatent2 * magnitude
                    fake_images_out = Gs.components.synthesis.get_output_for(dlatent_intepolate, randomize_noise=False)
                    pl_grads = tf.gradients(tf.reduce_sum(fake_images_out), [dlatent1])[0] / (1 - magnitude)
                    pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
                    y.append(pl_lengths[0])

                y_tf = tf.stack(y)
                for j in range(1, self.interpolation_steps - 1):
                    x = j / self.interpolation_steps
                    f = y_tf[0] + x * (y_tf[-1] - y_tf[0]) / ((self.interpolation_steps - 1) / self.interpolation_steps)
                    linear_diff += tf.cast(tf.square(y_tf[j] - f) / tf.abs(tf.reduce_max(y_tf) - tf.reduce_min(y_tf)), dtype=tf.float16).eval()

            # self._report_progress(i, self.num_images)
            result = linear_diff / self.num_images
            self._report_result(result)
            print(result)

    @staticmethod
    def _label_vector(model, color, manufacturer, body, rotation, ratio, background):
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

    def _get_label_one_diff(self):
        rotation_order = [0, 1, 3, 6, 5, 7, 4, 2]
        rotation = np.random.randint(8)
        model = np.random.randint(67)
        color = np.random.randint(12)
        body = np.random.randint(10)
        manufacturer = np.random.randint(18)
        ratio = np.random.randint(5)
        background = np.random.choice([0, 5])

        change = np.zeros(7, dtype=np.uint8)
        change[np.random.randint(6)] = 1

        label1 = self._label_vector(
            model=model,
            color=color,
            manufacturer=manufacturer,
            body=body,
            rotation=rotation_order[rotation],
            ratio=ratio,
            background=background
        )

        label2 = self._label_vector(
            model=(model + change[0]) % 67,
            color=(color + change[1]) % 12,
            manufacturer=(manufacturer + change[2]) % 18,
            body=(body + change[3]) % 10,
            rotation=rotation_order[(rotation + change[4]) % 8],
            ratio=(ratio + change[5]) % 5,
            background=(background + change[6]) % 6
        )

        return np.array([label1, label2])
