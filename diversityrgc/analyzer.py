# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from . import ovndata
from . import neuralnet
from . import log

# Logger
logger = log.get_logger(__name__)


def analyze(**kwargs):
    out_folder = kwargs['output_folder']

    # Load data
    stim, sv = ovndata.load_data(**kwargs)

    # Convolutional neural network model for video classification trained on the Kinetics dataset
    net = neuralnet.NeuralNet(temporal_size=kwargs['temporal_size'],
                              image_size=kwargs['neural_net_image_size'],
                              output_folder=out_folder)
    net.run(stim, sv, nspikes=1000, save=True)

    # Analyze data
    files = [f for f in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, f)) and f.endswith('.npy')]

    # Accumulated Pearson correlation coefficient
    logger.info('Calculate accumulated Pearson correlation coefficient')
    pcc = calculate_correlation(files, out_folder)

    logger.info('Plot Pearson correlation coefficient')
    xaxis = np.arange(pcc.size)
    plt.bar(xaxis, pcc)
    plt.title('Pearson correlation coefficient')
    plt.show()


def calculate_correlation(files, out_folder):
    pcc_size = np.load(out_folder + files[0]).shape[-1]
    pcc = np.zeros(pcc_size)
    counter = 0

    for i in range(len(files)):
        filters_x = np.load(out_folder + files[i])

        for j in range(i + 1, len(files)):
            filters_y = np.load(out_folder + files[j])

            for k in range(filters_x.shape[-1]):
                x = filters_x[:, :, :, k].flatten()
                y = filters_y[:, :, :, k].flatten()
                pcc[k] += pearsoncc(x, y)

            counter += 1

    return pcc / counter


def pearsoncc(x, y):
    cc = stats.pearsonr(x, y)[0]
    return 0.0 if np.isnan(cc) else cc
