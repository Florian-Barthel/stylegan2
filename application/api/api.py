from flask import Flask, render_template, request, jsonify
import sys
import numpy as np
from flask_cors import CORS
import application.api.generate_label as generate_label
import application.api.generate as generate

sys.path.append("../../")
app = Flask(__name__)
CORS(app)
generator = generate.Generate()
generator.load_network('results/00092-stylegan2-cars_v4_512-4gpu-config-f/network-snapshot-001925.pkl')


@app.route('/image', methods=['GET', 'POST'])
def image():
    data = request.get_json()
    payload = data['payload']
    randomize_seed = payload['randomize_seed']
    seed = int(payload['seed'])
    size = int(payload['size'])
    label = generate_label.label_vector(payload)
    if randomize_seed:
        seed = int(np.random.uniform() * (2 ** 32 - 1))
    result = generator.generate_single_image(label, seed, size)
    return jsonify({'status': result, 'seed': seed})


@app.route('/load_3d_view', methods=['POST'])
def load_3d_view():
    data = request.get_json()
    payload = data['payload']
    seed = int(payload['seed'])
    randomize_seed = payload['randomize_seed']
    if randomize_seed:
        seed = int(np.random.uniform() * (2 ** 32 - 1))
    labels = generate_label.label_vector_rotation(payload)
    result, cache_size = generator.rotations(labels=labels, seed=seed)
    return jsonify({'status': result, 'cache_size': cache_size, 'seed': seed})


@app.route('/load_interpolations', methods=['POST'])
def load_interpolations():
    data = request.get_json()
    payload_left = data['payload_left']
    payload_right = data['payload_right']
    seed_left = int(payload_left['seed'])
    seed_right = int(payload_right['seed'])
    label_left = generate_label.label_vector(payload_left)
    label_right = generate_label.label_vector(payload_right)
    result, cache_size = generator.interpolations(label_left=label_left,
                                                  seed_left=seed_left,
                                                  label_right=label_right,
                                                  seed_right=seed_right)
    return jsonify({'status': result, 'cache_size': cache_size})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/interpolate')
def interpolate():
    return render_template('interpolate.html')


@app.route('/3d_view')
def rotation():
    return render_template('3d_view.html')


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
    app.run(debug=False)
    # app.run(host='0.0.0.0', debug=False)
