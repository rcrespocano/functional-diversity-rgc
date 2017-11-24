# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.stats as stats
import scipy.misc as misc
import matplotlib.pyplot as plt
from . import ovndata
from . import neuralnet
from . import log
from . import io_utils

# Logger
logger = log.get_logger(__name__)


def analyze(**kwargs):
    # Convolutional neural network model for video classification trained on the Kinetics dataset
    net = neuralnet.NeuralNet(temporal_size=kwargs['temporal_size'], image_size=kwargs['neural_net_image_size'])
    output_folder_root = kwargs['output_folder']

    for cell in kwargs['cells']:
        out_folder = output_folder_root + 'cell_' + str(cell[0]) + str(cell[1]) + '/'
        io_utils.create_folder(out_folder)

        # Load data
        kwargs['cell_type_index'] = cell[0]
        kwargs['cell_number'] = cell[1]
        kwargs['output_folder'] = out_folder
        stim, sv = ovndata.load_data(**kwargs)

        # Run neural network
        net.run(stim, sv, out_folder, nspikes=kwargs['number_of_spikes'], save=True)

        # Analyze data
        files = [f for f in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, f)) and f.endswith('.npy')]
        files.sort(key=str.lower)

        # Accumulated Pearson correlation coefficient
        logger.info('Calculate accumulated Pearson correlation coefficient')
        pcc = calculate_correlation(files, out_folder)

        # Save plot of the accumulated Pearson correlation coefficient
        logger.info('Save plot of Pearson correlation coefficient')
        save_plot_pearsoncc(pcc, out_folder)

        # Show pcc > 0.75
        pcc_indexes = np.where(pcc > 0.75)[0]
        logger.info('Pearson correlation coefficient > 0.75')
        logger.info(pcc_indexes)

        # Save layer output of pcc > 0.75
        logger.info('Save layers as gif files with Pearson correlation coefficient > 0.75')
        net.run(stim, sv, out_folder, nspikes=1, save=False, gif=True, gif_indexes=pcc_indexes)

        # Run default stimulus on Kinetics I3D model to save the same filters
        logger.info('Run default stimulus on Kinetics I3D model to save the same filters')
        net.run_default_stim(gif_indexes=pcc_indexes, output_folder=out_folder)

        del stim, sv


def calculate_correlation(files, out_folder):
    pcc_size = np.load(out_folder + files[0]).shape[-1]
    pcc = np.zeros(pcc_size)
    counter = 0
    num_combinations = calculate_combinations(len(files))

    for i in range(len(files)):
        filters_x = np.load(out_folder + files[i])

        for j in range(i + 1, len(files)):
            filters_y = np.load(out_folder + files[j])
            counter += 1
            io_utils.print_progress_bar(counter, num_combinations, prefix='Progress:', suffix='Complete', length=50)

            for k in range(filters_x.shape[-1]):
                x = filters_x[:, :, :, k].flatten()
                y = filters_y[:, :, :, k].flatten()
                pcc[k] += pearsoncc(x, y)

    return pcc / counter


def save_plot_pearsoncc(pcc, output_folder):
    x_axis = np.arange(pcc.size)
    plt.bar(x_axis, pcc)
    plt.title('Pearson correlation coefficient')
    plt.savefig(output_folder + 'pcc.pdf')
    plt.clf()


def pearsoncc(x, y):
    cc = stats.pearsonr(x, y)[0]
    return 0.0 if np.isnan(cc) else cc


def calculate_combinations(n_elements):
    return misc.comb(n_elements, 2)
