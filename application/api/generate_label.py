import numpy as np


def label_vector_rotation(payload):
    manufacturer = int(payload['manufacturer'])
    model = int(payload['model'])
    color = int(payload['color'])
    body = int(payload['body'])
    ratio = int(payload['ratio'])
    background = int(payload['background'])

    onehot = np.zeros((8, 1, 127), dtype=np.float32)
    onehot[:, 0, 0] = 1
    offset = 1
    if model >= 0:
        onehot[:, 0, offset + model] = 1.0
    offset += 67
    if color >= 0:
        onehot[:, 0, offset + color] = 1.0
    offset += 12
    if manufacturer >= 0:
        onehot[:, 0, offset + manufacturer] = 1.0
    offset += 18
    if body >= 0:
        onehot[:, 0, offset + body] = 1.0
    offset += 10
    onehot[0, 0, offset + 0] = 1.0
    onehot[1, 0, offset + 1] = 1.0
    onehot[2, 0, offset + 3] = 1.0
    onehot[3, 0, offset + 6] = 1.0
    onehot[4, 0, offset + 5] = 1.0
    onehot[5, 0, offset + 7] = 1.0
    onehot[6, 0, offset + 4] = 1.0
    onehot[7, 0, offset + 2] = 1.0
    offset += 8
    onehot[:, 0, offset + ratio] = 1.0
    offset += 5
    if background >= 0:
        onehot[:, 0, offset + background] = 1.0
    return onehot


def label_vector(payload):
    manufacturer = int(payload['manufacturer'])
    model = int(payload['model'])
    color = int(payload['color'])
    body = int(payload['body'])
    rotation = int(payload['rotation'])
    ratio = int(payload['ratio'])
    background = int(payload['background'])

    onehot = np.zeros((1, 127), dtype=np.float32)
    onehot[0, 0] = 1
    offset = 1
    if model >= 0:
        onehot[0, offset + model] = 1.0
    offset += 67
    if color >= 0:
        onehot[0, offset + color] = 1.0
    offset += 12
    if manufacturer >= 0:
        onehot[0, offset + manufacturer] = 1.0
    offset += 18
    if body >= 0:
        onehot[0, offset + body] = 1.0
    offset += 10
    if rotation >= 0:
        onehot[0, offset + rotation] = 1.0
    offset += 8
    onehot[0, offset + ratio] = 1.0
    offset += 5
    if background >= 0:
        onehot[0, offset + background] = 1.0
    return onehot