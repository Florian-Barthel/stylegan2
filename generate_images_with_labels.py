import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
import training.misc as misc


def generate_images_with_labels(filename, Gs, w, h, num_labels, latents, truncation):
    canvas = PIL.Image.new('RGB', (w * num_labels, h * latents.shape[0]), 'white')
    for i in range(latents.shape[0]):
        for j in range(num_labels):
            onehot = np.zeros((1, num_labels), dtype=np.float)
            onehot[0, j] = 1.0
            image = Gs.get_output_for(latents[i],
                                      onehot,
                                      is_validation=True,
                                      randomize_noise=False,
                                      truncation_psi_val=truncation)
            image = tflib.convert_images_to_uint8(image, nchw_to_nhwc=True).eval()
            canvas.paste(PIL.Image.fromarray(image[0], 'RGB'), (j * w, i * h))
    canvas.save(filename)


def generate_images_color_labels(filename, Gs, w, h, num_labels, latents, truncation):
    canvas = PIL.Image.new('RGB', (w * 12, h * latents.shape[0]), 'white')
    for i in range(latents.shape[0]):
        for j in range(12):
            onehot = np.zeros((1, num_labels), dtype=np.float)
            onehot[0, 13] = 1.0
            onehot[0, j + 67] = 1.0
            onehot[0, 67 + 12 + 1] = 1.0
            image = Gs.get_output_for(latents[i],
                                      onehot,
                                      is_validation=True,
                                      randomize_noise=False,
                                      truncation_psi_val=truncation)
            image = tflib.convert_images_to_uint8(image, nchw_to_nhwc=True).eval()
            canvas.paste(PIL.Image.fromarray(image[0], 'RGB'), (j * w, i * h))
    canvas.save(filename)


def generate_images_manufacturer_labels(filename, Gs, w, h, num_labels, latents, truncation):
    canvas = PIL.Image.new('RGB', (w * 18, h * latents.shape[0]), 'white')
    for i in range(latents.shape[0]):
        for j in range(18):
            onehot = np.zeros((1, num_labels), dtype=np.float)
            onehot[0, 0 + 67] = 1.0
            onehot[0, 13] = 1.0
            onehot[0, 67 + 12 + j] = 1.0
            image = Gs.get_output_for(latents[i],
                                      onehot,
                                      is_validation=True,
                                      randomize_noise=False,
                                      truncation_psi_val=truncation)
            image = tflib.convert_images_to_uint8(image, nchw_to_nhwc=True).eval()
            canvas.paste(PIL.Image.fromarray(image[0], 'RGB'), (j * w, i * h))
    canvas.save(filename)


if __name__ == "__main__":
    tflib.init_tf()
    network_pkl = 'results/00021-stylegan2-all_cars_all_labels_256-2gpu-config-f/network-snapshot-009354.pkl'
    _G, _D, Gs = misc.load_pkl(network_pkl)
    latents = np.random.normal(0, 1, [12, 1, Gs.input_shape[1]])
    generate_images_color_labels('test_labels/colors-009354-1.0.png', Gs, w=256, h=256, num_labels=97, latents=latents,
                                 truncation=1.0)
