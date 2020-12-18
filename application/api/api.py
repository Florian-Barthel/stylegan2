from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import os, io, sys
import numpy as np
import base64
sys.path.append("../../")
import dnnlib.tflib as tflib
import training.misc as misc
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


class Graph:
    def __init__(self, network_pkl):
        tflib.init_tf()
        _G, _D, Gs = misc.load_pkl(network_pkl)
        self.network_pkl = network_pkl
        self.Gs = Gs

    def generate_image(self, onehot, seed):
        tflib.init_tf()
        _G, _D, Gs = misc.load_pkl(self.network_pkl)
        self.Gs = Gs
        if seed is -1:
            latents = np.random.normal(0, 1, [1, self.Gs.input_shape[1]])
        else:
            latents = np.random.RandomState(seed).normal(0, 1, [1, self.Gs.input_shape[1]])
        image = self.Gs.get_output_for(latents, onehot, is_validation=True, randomize_noise=False, truncation_psi_val=0.7)
        return tflib.convert_images_to_uint8(image, nchw_to_nhwc=True).eval()[0]


graph = Graph('../../results/00021-stylegan2-all_cars_all_labels_256-2gpu-config-f/network-snapshot-004112.pkl')


@app.route('/image', methods=['GET', 'POST'])
def image():
    data = str(request.data)[1:].replace('\'', '').split(',')
    print(data)
    manufacturer = int(data[0])
    model = int(data[1])
    color = int(data[2])
    seed = int(data[3])

    onehot = np.zeros((1, 97), dtype=np.float)
    onehot[0, model] = 1.0
    onehot[0, color + 67] = 1.0
    onehot[0, 67 + 12 + manufacturer] = 1.0

    image = graph.generate_image(onehot, seed)
    img = image[48:-48,:]
    img = Image.fromarray(img.astype("uint8"))
    img = img.resize((256 * 3, 160 * 3))

    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status': str(img_base64)})


@app.route('/')
def index():
    return render_template('index.html')


@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
    app.run(debug=True)