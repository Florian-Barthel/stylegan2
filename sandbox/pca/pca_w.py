import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from application.api.generate import Generate
from application.api.classes import classes

generator = Generate()
# generator.load_network('00070-stylegan2-cars_v4_512-2gpu-config-f/network-snapshot-007219.pkl') # separate label mapping add
# generator.load_network('00102-stylegan2-cars_v4_512-2gpu-config-f/network-snapshot-006437.pkl') # separate label mapping concat
generator.load_network('00037-stylegan2-cars_v4_512-2gpu-config-f-baseline/network-snapshot-006316.pkl') # baseline


n_components = 2
pca = PCA(n_components=n_components)


def label_vector(model, color, manufacturer, body, rotation, ratio, background):
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


def write_to_file(filename, strings):
    file = open(filename, 'a')
    for s in strings:
        file.write(str(s))
        file.write('\t')
    file.write('\n')
    file.close()


def init_file(filename):
    file = open(filename, 'w')
    file.write('')
    file.close()

label_filename = 'labels_rotation_baseline_int.tsv'
init_file(label_filename)

write_to_file(label_filename, ['color'])
X = []
images = []
interpolate = True
for i in range(500):
    #color = np.random.randint(0, 12)
    rotation = np.random.randint(0, 8)
    label = label_vector(0, 0, 0, 0, rotation, 1, 0)
    if interpolate:
        label_int = label_vector(0, 0, 0, 0, (rotation + np.random.choice([-1, 1])) % 8, 1, 0)
        mag = np.random.uniform()
        label = label * mag + label_int * (1 - mag)
    seed = 3 # np.random.randint(0, 100000)
    image, dlatent = generator.generate_single_image(label=label, seed=seed, size=256, convert_to_base64_str=False, return_dlatent=True)
    X.append(dlatent[0, 0, :])
    images.append(np.asarray(image))
    write_to_file(label_filename, [classes[4]['classes'][rotation]])

vec_filename = 'vectors_rotation_baseline_int.tsv'
init_file(vec_filename)

for i in range(len(X)):
    write_to_file(vec_filename, X[i])

X_pca = pca.fit_transform(X)

x = X_pca[:, 0]
y = X_pca[:, 1]

fig, ax = plt.subplots(dpi=500)
plt.axis('off')
ax.scatter(x, y)

for x0, y0, image in zip(x, y, images):
    ab = AnnotationBbox(OffsetImage(image, zoom=0.05), (x0, y0), frameon=False)
    ax.add_artist(ab)

# plt.plot(X_pca)
plt.savefig('pca_rotation_baseline_int.png')
plt.show()

