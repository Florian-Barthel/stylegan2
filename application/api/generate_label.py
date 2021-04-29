import numpy as np


def label_vector_rotation(payload):
    rotation_order = [0, 1, 3, 6, 5, 7, 4, 2]
    label = np.zeros((8, 1, 127), dtype=np.float32)
    for i in range(8):
        payload['rotation'] = rotation_order[i]
        label[i] = label_vector(payload)
    return label


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
