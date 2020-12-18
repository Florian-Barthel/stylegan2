
import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
import training.misc as misc


synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)



def draw_style_mixing_figure_transition(png, Gs, w, h, style1_seeds, style2_seed, style_ranges):
    style1_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in style1_seeds)
    style2_latents = np.stack(
        np.random.RandomState(style2_seed).randn(Gs.input_shape[1]) for _ in range(len(style_ranges)))
    style1_dlatents = Gs.components.mapping.run(style1_latents, None)  # [seed, layer, component]
    style2_dlatents = Gs.components.mapping.run(style2_latents, None)  # [seed, layer, component]
    style1_images = Gs.components.synthesis.run(style1_dlatents, randomize_noise=False, **synthesis_kwargs)
    style2_image = Gs.components.synthesis.run(style2_dlatents, randomize_noise=False, **synthesis_kwargs)[0]

    canvas = PIL.Image.new('RGB', (w * (len(style_ranges) + 1), h * (len(style1_seeds) + 1)), 'white')

    for row, src_image in enumerate(list(style1_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), (0, (row + 1) * h))
    for row in range(len(style_ranges)):
        canvas.paste(PIL.Image.fromarray(style2_image, 'RGB'), ((row + 1) * h, 0))
        for col in range(len(style1_seeds)):
            mixed_dlatent = np.array([style1_dlatents[col]])
            mixed_dlatent[:, style_ranges[row], :] = style2_dlatents[row, style_ranges[row], :]
            image = Gs.components.synthesis.run(mixed_dlatent, randomize_noise=False, **synthesis_kwargs)[0]
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((row + 1) * w, (col + 1) * h))
    canvas.save(png)
    print(png)


def generate_images_with_labels(filename, Gs, w, h, num_labels, num_seeds, layer):
    latents = np.random.normal(0, 1, [num_seeds, 1, Gs.input_shape[1]])
    color_latent = np.random.normal(0, 1, [num_labels, 1, Gs.input_shape[1]])
    canvas = PIL.Image.new('RGB', (w * num_labels, h * num_seeds), 'white')
    for i in range(num_seeds):
        print(i)
        for j in range(num_labels):
            onehot = np.zeros((1, num_labels), dtype=np.float)
            onehot[0, 0] = 1.0

            onehot_color = np.zeros((1, num_labels), dtype=np.float)
            onehot_color[0, j] = 1.0

            #image = Gs.get_output_for(latents[i], onehot, is_validation=True, randomize_noise=False, truncation_psi_val=1.0)

            dlatent = Gs.components.mapping.run(latents[i], onehot)
            dlatent_color = Gs.components.mapping.run(color_latent[j], onehot_color)

            mixed_dlatent = np.array(dlatent)
            mixed_dlatent[:, layer, :] = dlatent_color[0, layer, :]

            # image = tflib.convert_images_to_uint8(image, nchw_to_nhwc=True).eval()
            image = Gs.components.synthesis.run(mixed_dlatent, randomize_noise=False, **synthesis_kwargs)[0]
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), (j * w, i * h))
    canvas.save(filename)


def main():
    tflib.init_tf()
    network_pkl = 'results/00014-stylegan2-cars_color_labels-2gpu-config-f/network-snapshot-003068.pkl'
    _G, _D, Gs = misc.load_pkl(network_pkl)

    draw_style_mixing_figure_transition('test_labels/style_mix_0.png', Gs, w=512, h=512, style1_seeds=[12, 2, 13], style2_seed=[116], style_ranges=[list(range(i-2, i)) for i in range(2, 18, 2)])

    # generate_images_with_labels('test_labels/colors3.png', Gs, w=512, h=512, num_labels=12, num_seeds=12, layer=4)


if __name__ == "__main__":
    main()
