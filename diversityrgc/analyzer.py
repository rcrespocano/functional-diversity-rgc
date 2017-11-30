# -*- coding: utf-8 -*-

import os
import random
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
    net = neuralnet.NeuralNet(temporal_size=kwargs['temporal_size'],
                              image_size=kwargs['neural_net_image_size'],
                              layer_name=kwargs['layer_name'],
                              layer_shape=kwargs['layer_shape'])
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
        net.run(stim, sv, out_folder, nspikes=kwargs['number_of_spikes'], start=0, save=True)
        '''
        # Analyze data
        files = [f for f in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, f))
                 and f.endswith('.npy') and 'mean' not in f]
        files.sort(key=str.lower)

        # Accumulated Pearson correlation coefficient
        logger.info('Calculate accumulated Pearson correlation coefficient')
        pcc = __calculate_correlation(files, out_folder)

        # Save plot of the accumulated Pearson correlation coefficient
        logger.info('Save plot of Pearson correlation coefficient')
        __save_plot_pearsoncc(pcc, out_folder)
        '''


def compare_correlated_filters(**kwargs):
    folder = kwargs['folder']
    output_folder = kwargs['output_folder']
    layer_shape = kwargs['layer_shape']
    cell_target = kwargs['cell_target']
    rows = 5
    cols = 5

    cell_folders = [x[0] for x in os.walk(folder) if 'cell' in x[0] and cell_target not in x[0]]
    cell_folders.sort(key=str.lower)
    cell_target_folder = [x[0] for x in os.walk(folder) if 'cell' in x[0] and cell_target in x[0]]
    num_cells = len(cell_folders) + 1
    num_combinations = __calculate_combinations(num_cells)
    logger.info('Cell folders')
    logger.info(cell_folders)
    logger.info('Cell target')
    logger.info(cell_target_folder)

    # Save correlation of all filters
    data = np.empty((num_cells,) + layer_shape)
    data[0] = np.load(cell_target_folder[0] + '/mean.npy')
    for i in range(1, num_cells):
        data[i] = np.load(cell_folders[i-1] + '/mean.npy')

    pcc = np.empty(data.shape[-1])
    for k in range(data.shape[-1]):
        _pcc = 0.0
        for n in range(num_cells):
            _x = data[n, :, k]
            for m in range(n+1, num_cells):
                _y = data[m, :, k]
                _pcc += __pearsoncc(_x, _y)

        pcc[k] = _pcc / num_combinations

    __save_plot_pearsoncc(pcc=pcc, output_folder=output_folder, name='pcc_all_filters.pdf')

    # Save correlation only with the target cell
    pcc_values = np.empty((num_cells-1,) + (data.shape[-1],))
    for n in range(1, num_cells):
        for k in range(data.shape[-1]):
            _x = data[0, :, k]
            _y = data[n, :, k]
            pcc_values[n-1][k] = __pearsoncc(data[0, :, k], data[n, :, k])

    logger.info('PCC less than 0.5')
    for i in range(len(pcc_values)):
        logger.info('Comparison %s', i)
        logger.info(np.where([pcc_values[i] < 0.5])[1])

    f, axarr = plt.subplots(rows, cols)
    for i in range(pcc_values.shape[0]):
        x_axis = np.arange(pcc_values[i].size)
        x = i // rows
        y = i % cols
        axarr[x, y].bar(x_axis, pcc_values[i])
    plt.savefig(output_folder + 'pcc_in_pairs.pdf')
    plt.clf()


def plot_filters(title=False, **kwargs):
    folder = kwargs['folder']
    output_folder = kwargs['output_folder']

    cell_folders = [x[0] for x in os.walk(folder) if 'cell' in x[0]]
    cell_folders.sort(key=str.lower)

    f, axarr = plt.subplots(nrows=len(cell_folders), ncols=1)

    for i, f in enumerate(cell_folders):
        logger.info('Load filter from [' + f + ']')
        data = np.load(f + '/mean.npy')
        if title:
            axarr[i].set_title(f)
        axarr[i].axis('off')
        axarr[i].imshow(data)

    plt.savefig(output_folder + 'filters_colormap.pdf')
    plt.clf()


def __calculate_correlation(files, out_folder):
    pcc_size = np.load(out_folder + files[0]).shape[-1]
    pcc = np.zeros(pcc_size)
    counter = 0
    num_combinations = __calculate_combinations(len(files))

    for i in range(len(files)):
        filters_x = np.load(out_folder + files[i])

        for j in range(i + 1, len(files)):
            filters_y = np.load(out_folder + files[j])
            io_utils.print_progress_bar(counter, num_combinations, prefix='Progress:', suffix='Complete', length=50)
            counter += 1

            for k in range(filters_x.shape[-1]):
                pcc[k] += __pearsoncc(filters_x[:, k], filters_y[:, k])

    return pcc / counter


def __save_plot_pearsoncc(pcc, output_folder, name='pcc.pdf'):
    x_axis = np.arange(pcc.size)
    plt.bar(x_axis, pcc)
    plt.title('Pearson correlation coefficient')
    plt.savefig(output_folder + name)
    plt.clf()


def __pearsoncc(x, y):
    cc = stats.pearsonr(x, y)[0]
    return 0.0 if np.isnan(cc) else cc


def __calculate_combinations(n_elements):
    return misc.comb(n_elements, 2)


def __old_deprecated_compare_correlated_filters(**kwargs):
    folder = kwargs['folder']
    layer_name = kwargs['layer_name']
    spikes = kwargs['spikes']
    filters = kwargs['filters']
    output_folder = kwargs['output_folder']

    cell_folders = [x[0] for x in os.walk(folder) if 'cell' in x[0]]
    cell_folders.sort(key=str.lower)
    num_combinations = __calculate_combinations(len(cell_folders))

    # Output: (num_filters, num_comparisons_filters_all_to_all)
    output = np.zeros((len(filters), int(num_combinations)))

    for index, filter in enumerate(filters):
        counter = 0

        for i in range(len(cell_folders)):
            ln = layer_name + '_' + (str(random.randint(0, spikes)).zfill(6)) + '.npy'
            _x = np.load(cell_folders[i] + '/' + ln)[:, :, :, int(filter)].flatten()

            for j in range(i + 1, len(cell_folders)):
                ln = layer_name + '_' + (str(random.randint(0, spikes)).zfill(6)) + '.npy'
                _y = np.load(cell_folders[j] + '/' + ln)[:, :, :, int(filter)].flatten()
                output[index][counter] = __pearsoncc(_x, _y)
                counter += 1

    # Save Pearson correlation coefficient for each filter comparison (in a different plot)
    for index, filter in enumerate(filters):
        __save_plot_pearsoncc(pcc=output[index], output_folder=output_folder, name='pcc_filter_'+str(filter)+'.pdf')

    return output
