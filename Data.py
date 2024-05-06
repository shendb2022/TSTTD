import matplotlib.pyplot as plt
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import torch
from ts_generation import ts_generation
from Tools import standard


class Data(data.Dataset):
    def __init__(self, path):
        mat = sio.loadmat(path)
        data = mat['data']
        gt = mat['map']
        data = standard(data)

        h, w, b = data.shape
        pixel_nums = h * w

        ## get the target spectrum
        target_spectrum = ts_generation(data, gt, 7)

        ## regard all the pixels as background pixels
        background_samples = np.reshape(data, [-1, b], order='F')

        ## randomly generate target samples by linear representation
        alphas = np.random.uniform(0, 0.1, pixel_nums)
        alphas = alphas[:, None]
        target_samples = alphas * background_samples + (1 - alphas) * target_spectrum.T

        self.target_samples = target_samples
        self.background_samples = background_samples
        self.target_spectrum = target_spectrum.T
        self.nums = pixel_nums

    def __getitem__(self, index):
        positive_samples = self.target_samples[index]
        negative_samples = self.background_samples[index]
        return positive_samples, negative_samples

    def __len__(self):
        return self.nums


if __name__ == '__main__':
    data = Data('/home/sdb/Codes/datasets2/Sandiego.mat')
    target_samples = data.target_spectrum
    print(target_samples.shape)
    plt.plot(target_samples.T)
    plt.show()
    # center, coded_vector = data.__getitem__(128)
    # plt.plot(center.T)
    # plt.plot(coded_vector.T)
    # plt.show()
