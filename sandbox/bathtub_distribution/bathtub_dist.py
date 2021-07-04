import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class BathtubDistribution(st.rv_continuous):
    def __init__(self, scale, **kwargs):
        super(BathtubDistribution, self).__init__(**kwargs)
        self.scale = scale

    def _pdf(self, x):
        # return ((x - 0.5)**2 * self.scale + 1) / (1 + self.scale / 12)
        # return ((x - 0.5)**2 * self.scale + 1) / (((self.scale*(1)**3)/24+1) - ((self.scale*(-1)**3)/24+0))
        zero_intersect = (self.scale * (-0.5)**4 + 1) / (self.scale / 80 + 1)
        a = zero_intersect * (self.scale / 80 + 1)
        return (a * (x - 0.5)**4 + 1) / (a / 80 + 1)


scales = [0, 1, 10, 100, 1000, 10000]
for scale in scales:
    bathtub = BathtubDistribution(a=0.0, b=1.0, name='my_pdf', scale=scale)
    y = np.zeros([1001])
    x = np.arange(0, 1 + 0.001, 0.001)
    for i in tqdm(range(50000)):
        y[int(bathtub.rvs().round(3) * 1000)] += 1
    plt.plot(x, y)
    plt.legend(str(scale))

plt.savefig('bathtub_dist.png')
plt.show()
