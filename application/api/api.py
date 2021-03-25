from flask import Flask, render_template, request, jsonify
import io, sys
import numpy as np
import base64
import application.api.generate as generate
from flask_cors import CORS

sys.path.append("../../")

app = Flask(__name__)
CORS(app)


@app.route('/image', methods=['GET', 'POST'])
def image():
    data = request.get_json()
    payload = data['payload']
    manufacturer = int(payload['manufacturer'])
    model = int(payload['model'])
    color = int(payload['color'])
    body = int(payload['body'])
    rotation = int(payload['rotation'])
    ratio = int(payload['ratio'])
    background = int(payload['background'])
    randomize_seed = payload['randomize_seed']
    seed = int(payload['seed'])
    size = int(payload['size'])

    label = generate.label_vector(model=model,
                                  color=color,
                                  manufacturer=manufacturer,
                                  body=body,
                                  rotation=rotation,
                                  ratio=ratio,
                                  background=background)

    if randomize_seed:
        seed = int(np.random.uniform() * (2 ** 32 - 1))
    image = generate.single_image(label, seed, size)
    rawBytes = io.BytesIO()
    image.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status': str(img_base64), 'seed': seed})


@app.route('/load_interpolations', methods=['POST'])
def load_interpolations():
    data = request.get_json()
    payload = data['payload']
    manufacturer_left = int(payload['manufacturer_left'])
    model_left = int(payload['model_left'])
    color_left = int(payload['color_left'])
    body_left = int(payload['body_left'])
    rotation_left = int(payload['rotation_left'])
    ratio_left = int(payload['ratio_left'])
    background_left = int(payload['background_left'])
    seed_left = int(payload['seed_left'])

    label_left = generate.label_vector(model=model_left,
                                       color=color_left,
                                       manufacturer=manufacturer_left,
                                       body=body_left,
                                       rotation=rotation_left,
                                       ratio=ratio_left,
                                       background=background_left)

    manufacturer_right = int(payload['manufacturer_right'])
    model_right = int(payload['model_right'])
    color_right = int(payload['color_right'])
    body_right = int(payload['body_right'])
    rotation_right = int(payload['rotation_right'])
    ratio_right = int(payload['ratio_right'])
    background_right = int(payload['background_right'])
    seed_right = int(payload['seed_right'])

    label_right = generate.label_vector(model=model_right,
                                        color=color_right,
                                        manufacturer=manufacturer_right,
                                        body=body_right,
                                        rotation=rotation_right,
                                        ratio=ratio_right,
                                        background=background_right)

    return generate.interpolations(label_left, seed_left, label_right, seed_right)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/interpolate')
def interpolate():
    return render_template('interpolate.html')


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
    app.run(debug=False)
    # app.run(host='0.0.0.0', debug=False)
