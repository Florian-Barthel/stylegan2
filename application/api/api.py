from flask import Flask, render_template, request, jsonify
from PIL import Image
import io, sys
import numpy as np
import base64
sys.path.append("../../")
CUDA_VISIBLE_DEVICES = ""

import dnnlib.tflib as tflib
import training.misc as misc
from flask_cors import CORS
import tensorflow as tf


app = Flask(__name__)
CORS(app)

session = tflib.create_session(None, force_as_default=True)

latent_placeholder = tf.placeholder(tf.float32, shape=(None, 512))
label_placeholder = tf.placeholder(tf.float32, shape=(None, 121))


_G, _D, Gs = misc.load_pkl('../../results/00032-stylegan2-cars_v2_512-2gpu-config-f/network-snapshot-001564.pkl')
gen_image = Gs.get_output_for(latent_placeholder, label_placeholder, is_validation=True, randomize_noise=True,
                          truncation_psi_val=1.0)


def run(label, seed):
    if seed is -1:
        latent = np.random.normal(0, 1, [1, Gs.input_shape[1]]).astype(np.float32)
    else:
        latent = np.random.RandomState(seed).normal(0, 1, [1, Gs.input_shape[1]]).astype(np.float32)

    with tf.device('/cpu:0'):
        image = session.run(gen_image, feed_dict={latent_placeholder: latent, label_placeholder: label})[0]
        image = np.transpose(image, [1, 2, 0])
        image = image * 127.5 + 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image


@app.route('/image', methods=['GET', 'POST'])
def image():
    data = request.get_json()
    # print(data)
    payload = data['payload']
    print(payload)
    manufacturer = int(payload['manufacturer'])
    model = int(payload['model'])
    color = int(payload['color'])
    body = int(payload['body'])
    rotation = int(payload['rotation'])
    ratio = int(payload['ratio'])
    seed = int(payload['seed'])

    onehot = np.zeros((1, 121), dtype=np.float)
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

    # print(onehot)
    image = run(onehot, seed)
    # img = image[48:-48,:]
    img = Image.fromarray(image)
    img = img.resize((512, 512))

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
    # print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
    #app.run(debug=False)
    app.run(host='0.0.0.0', debug=False)