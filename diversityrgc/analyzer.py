# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import scipy.stats as stats
import scipy.misc as misc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from . import ovndata
from . import neuralnet
from . import log
from . import io_utils
from . import filtertools

# Logger
logger = log.get_logger(__name__)


def analyze(analyze_pcc=False, **kwargs):
    # Convolutional neural network model for video classification trained on the Kinetics dataset
    net = neuralnet.NeuralNet(temporal_size=kwargs['temporal_size'],
                              image_size=kwargs['neural_net_image_size'],
                              layer_names=kwargs['layer_names'],
                              layer_shapes=kwargs['layer_shapes'],
                              layer_sizes=kwargs['layer_sizes'])
    output_folder_root = kwargs['output_folder']

    for cell in kwargs['cells']:
        out_folder = output_folder_root + 'cell_' + str(cell[0]) + str(cell[1]) + '/'
        io_utils.create_folder(out_folder)

        # Load data
        kwargs['cell_type_index'] = cell[0]
        kwargs['cell_number'] = cell[1]
        kwargs['output_folder'] = out_folder

        # Load spike vector
        sv, mean = ovndata.load_spike_vector(**kwargs)
        kwargs['sv'] = sv
        kwargs['mean'] = mean

        # Load stim
        stim = ovndata.load_stim(**kwargs)

        # Run neural network
        net.run(stim, sv, out_folder, nspikes=kwargs['number_of_spikes'], start=0, save=True)

        if analyze_pcc:
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

        del sv, mean, stim


def compare_correlated_feature_maps(**kwargs):
    root_folder = kwargs['folder']
    output_folder = kwargs['output_folder']
    layer_names = kwargs['layer_names']
    layer_shapes = kwargs['layer_shapes']

    cell_folders = [x[0] for x in os.walk(root_folder) if 'cell' in x[0]]

    for folder in cell_folders:
        logger.info('Folder: %s', folder)

        # Files
        _files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.npy')]
        _files.sort(key=str.lower)

        for _idx, _layer in enumerate(layer_names):
            _feature_maps = [x for x in _files if _layer in x]
            _feature_maps.sort(key=str.lower)
            _n_feature_maps = len(_feature_maps)
            _ncombinations = __calculate_combinations(_n_feature_maps)
            _pcc = np.zeros(layer_shapes[_idx][-1])
            _counter = 0

            for i in range(_n_feature_maps):
                _data_i = np.load(folder + '/' + _feature_maps[i])[0, :, :, :, :]
                for j in range(i+1, _n_feature_maps):
                    _data_j = np.load(folder + '/' + _feature_maps[j])[0, :, :, :, :]
                    _counter += 1
                    io_utils.print_progress_bar(_counter, _ncombinations, prefix='Progress:', suffix='Complete', length=50)

                    for k in range(layer_shapes[_idx][-1]):
                        _pcc[k] += __pearsoncc(_data_i[:, :, :, k].flatten(), _data_j[:, :, :, k].flatten())

            _pcc /= _ncombinations
            __save_plot_pearsoncc(pcc=_pcc, output_folder=output_folder,
                                  name='pcc_' + os.path.basename(folder) + '_' + _layer + '.pdf')


def analyze_principal_components(name=None, **kwargs):
    root_folder = kwargs['folder']
    layer_names = kwargs['layer_names']
    output_folder = kwargs['output_folder']

    cell_folders = [x[0] for x in os.walk(root_folder) if 'cell' in x[0]]

    for _idx, folder in enumerate(cell_folders):
        logger.info('Folder: %s', folder)

        for layer in layer_names:
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.npy')
                     and 'mean' not in f and 'sta' not in f and layer in f]
            files.sort(key=str.lower)

            # Number of feature maps
            layer_feature_maps = 0
            if len(files) > 0:
                data = np.load(folder + '/' + files[0])
                layer_feature_maps = data.shape[-1]

            for j in range(layer_feature_maps):
                for i, f in enumerate(files):
                    try:
                        data = np.load(folder + '/' + f)
                        data = data[0, :, :, :, j]
                        data_norm = (data - data.min()) / (data.max() - data.min())
                        data_norm = data_norm.reshape((data_norm.shape[0], -1))

                        # PCA
                        pca = PCA(n_components=2)
                        projected = pca.fit_transform(data_norm)
                        plt.scatter(projected[:, 0], projected[:, 1])
                    except:
                        pass

                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.title(folder)
                plt.savefig(output_folder + 'pca_' + os.path.basename(folder) + '_' + layer + '_'+ str(j) + '.png')
                plt.clf()


def analyze_principal_components_space_reduction(name=None, **kwargs):
    root_folder = kwargs['folder']
    output_folder = kwargs['output_folder']
    layer_names = kwargs['layer_names']
    layer_shapes = kwargs['layer_shapes']
    layer_centers = kwargs['layer_centers']
    _plot_pca = True
    '''
    _group_limit = 15
    rows = 3
    cols = 5
    '''

    cell_folders = [x[0] for x in os.walk(root_folder) if 'cell' in x[0]]

    for _idx, folder in enumerate(cell_folders):
        logger.info('Folder: %s', folder)

        # Files
        _files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.npy')]
        _files.sort(key=str.lower)

        for _idx, _layer in enumerate(layer_names):
            logger.info('> Layer %s', _layer)
            _feature_maps = [x for x in _files if _layer in x]
            _feature_maps.sort(key=str.lower)
            _n = len(_feature_maps)
            _all_fm = np.zeros((_n, layer_shapes[_idx][0] * layer_shapes[_idx][3]))
            '''
            # _sta_group_a = np.zeros((layer_shapes[_idx][0], layer_shapes[_idx][1], layer_shapes[_idx][2]))
            # _sta_group_b = np.zeros((layer_shapes[_idx][0], layer_shapes[_idx][1], layer_shapes[_idx][2]))
            # _group_counters = [0, 0]
            '''

            for _i, _fm in enumerate(_feature_maps):
                io_utils.print_progress_bar(_i, _n, prefix='Progress:', suffix='Complete', length=50)

                # Load feature maps from single spike
                _data = np.load(folder + '/' + _fm)
                _reduced_data = _data[0, :, layer_centers[_idx][0]:layer_centers[_idx][1]+1,
                                      layer_centers[_idx][0]:layer_centers[_idx][1] + 1, :]
                _reduced_data = np.mean(_reduced_data, axis=(1, 2))
                _all_fm[_i] = _reduced_data.flatten()

                '''
                # Calculate the STA for the two groups
                # group_a_idx = np.where(projected[:, 0] < _group_limit)[0]
                # group_b_idx = np.where(projected[:, 0] >= _group_limit)[0]

                for _index in group_a_idx:
                    _sta_group_a += _data[0, :, :, :, _index]
                    _group_counters[0] += 1

                for _index in group_b_idx:
                    _sta_group_b += _data[0, :, :, :, _index]
                    _group_counters[1] += 1
                '''

            if _plot_pca:
                # Calculate PCA
                pca = PCA(n_components=2)
                projected = pca.fit_transform(_all_fm)
                plt.scatter(projected[:, 0], projected[:, 1])
                plt.title('[PCA] ' + _layer)
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.savefig(output_folder + 'pca_' + _layer + '.pdf')
                plt.clf()

            '''
            # STA
            _sta_group_a /= _group_counters[0]
            _sta_group_b /= _group_counters[1]

            f, axarr = plt.subplots(rows, cols)
            for i in range(_sta_group_a.shape[0]):
                x = i // cols
                y = i % cols
                axarr[x, y].imshow(_sta_group_a[i])

            plt.savefig(output_folder + 'sta-' + _layer + '-' + 'a' + '.pdf')
            plt.clf()
            f, axarr = plt.subplots(rows, cols)
            for i in range(_sta_group_b.shape[0]):
                x = i // cols
                y = i % cols
                axarr[x, y].imshow(_sta_group_b[i])

            plt.savefig(output_folder + 'sta-' + _layer + '-' + 'b' + '.pdf')
            plt.clf()
            '''


def save_sta_graphs(**kwargs):
    logger.info('Save feature maps STA decomposed')
    save_feature_maps_sta_decomposed(**kwargs)

    logger.info('Save feature maps STA decomposed')
    save_feature_maps_sta_images(**kwargs)


def save_feature_maps_sta_decomposed(**kwargs):
    root_folder = kwargs['folder']
    layer_names = kwargs['layer_names']
    output_folder = kwargs['output_folder']

    cell_folders = [x[0] for x in os.walk(root_folder) if 'cell' in x[0]]

    for _idx, folder in enumerate(cell_folders):
        logger.info('Folder: %s', folder)

        try:
            for layer in layer_names:
                file = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.npy')
                         and 'sta' in f and layer in f]
                name, _ = os.path.splitext(file[0])
                base = os.path.basename(os.path.normpath(folder))

                # Load data
                data = np.load(folder + '/' + file[0])
                for j in range(data.shape[-1]):
                    spatial_kernel, temporal_kernel = filtertools.decompose_sta(data[0, :, :, :, j])

                    # Two subplots, the axes array is 1-d
                    f, axarr = plt.subplots(2)
                    axarr[0].set_title('STA Spatial Kernel')
                    axarr[0].imshow(spatial_kernel)
                    axarr[1].set_title('STA Temporal Kernel')
                    axarr[1].plot(np.arange(-data.shape[1] + 1, 1), temporal_kernel)
                    f.subplots_adjust(hspace=0.5)
                    plt.savefig(output_folder + base + '-' + name + '_dec_' + str(j) + '_' + '.png')
                    plt.clf()
                    plt.close('all')
        except:
            pass


def save_feature_maps_sta_images(**kwargs):
    root_folder = kwargs['folder']
    layer_names = kwargs['layer_names']
    output_folder = kwargs['output_folder']
    rows = 4
    cols = 4

    cell_folders = [x[0] for x in os.walk(root_folder) if 'cell' in x[0]]

    for _idx, folder in enumerate(cell_folders):
        logger.info('Folder: %s', folder)

        try:
            for layer in layer_names:
                file = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.npy')
                         and '_sta' in f and layer in f]
                name, _ = os.path.splitext(file[0])
                base = os.path.basename(os.path.normpath(folder))

                # Load data
                data = np.load(folder + '/' + file[0])

                for j in range(data.shape[-1]):
                    _sta = data[0, :, :, :, j]

                    f, axarr = plt.subplots(rows, cols)
                    for i in range(_sta.shape[0]):
                        x = i // cols
                        y = i % cols
                        axarr[x, y].imshow(_sta[i])

                    plt.savefig(output_folder + base + '-' + name + '_' + str(j) + '_' + '.png')
                    plt.clf()
                    plt.close('all')
        except:
            pass


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


def _deprecated_plot_filters(title=True, **kwargs):
    folder = kwargs['folder']
    output_folder = kwargs['output_folder']
    layer_shapes = kwargs['layer_shapes']

    cell_folders = [x[0] for x in os.walk(folder) if 'cell' in x[0]]
    cell_folders.sort(key=str.lower)

    for _idx, _ in enumerate(layer_shapes):
        f, axarr = plt.subplots(nrows=len(cell_folders), ncols=1)
        for i, f in enumerate(cell_folders):
            filename = f + '/mean_' + str(_idx) + '.npy'
            data = np.load(filename)
            logger.info('> Load filename %s', filename)

            if title:
                axarr[i].set_title(f)
            axarr[i].axis('off')
            axarr[i].imshow(data)
            plt.imshow(data)

        plt.savefig(output_folder + 'filters_colormap_' + str(_idx) + '.pdf')
        plt.clf()


def _deprecated_compare_correlated_filters(**kwargs):
    folder = kwargs['folder']
    output_folder = kwargs['output_folder']
    layer_shapes = kwargs['layer_shapes']
    cell_target = kwargs['cell_target']
    rows = 2
    cols = 2

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
    logger.info('Save correlation of all filters')
    for _idx, layer_shape in enumerate(layer_shapes):
        mean_file = 'mean_' + str(_idx) + '.npy'
        logger.info('Mean file %s', mean_file)

        data = np.empty((num_cells,) + layer_shape)
        data[0] = np.load(cell_target_folder[0] + '/' + mean_file)
        for i in range(1, num_cells):
            logger.info('> Data i=' + str(i) + ' --> ' + cell_folders[i-1])
            data[i] = np.load(cell_folders[i-1] + '/' + mean_file)

        pcc = np.empty(data.shape[-1])
        for k in range(data.shape[-1]):
            _pcc = 0.0
            for n in range(num_cells):
                _x = data[n, :, k]
                for m in range(n+1, num_cells):
                    _y = data[m, :, k]
                    _pcc += __pearsoncc(_x, _y)

            pcc[k] = _pcc / num_combinations

        __save_plot_pearsoncc(pcc=pcc, output_folder=output_folder, name='pcc_all_filters_' + str(_idx) + '.pdf')

        # Save correlation only with the target cell
        logger.info('Save correlation only with the target cell')
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
            x = i // cols
            y = i % cols
            axarr[x, y].bar(x_axis, pcc_values[i])
        plt.savefig(output_folder + 'pcc_in_pairs_' + str(_idx) + '.pdf')
        plt.clf()


def __deprecated_compare_correlated_filters_old(**kwargs):
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
        __save_plot_pearsoncc(pcc=output[index], output_folder=output_folder, name='pcc_filter_' + str(filter) + '.pdf')

    return output


def __deprecated_save_feature_maps_sta_decomposed(**kwargs):
    root_folder = kwargs['folder']
    layer_name = kwargs['layer_name']
    output_folder = kwargs['output_folder']

    cell_folders = [x[0] for x in os.walk(root_folder) if 'cell' in x[0]]
    num_feat_maps = 64
    temp_size = 15
    size = 112

    for _idx, folder in enumerate(cell_folders):
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.npy')
                 and 'mean' not in f and layer_name in f]
        files.sort(key=str.lower)
        num_files = len(files)

        for j in range(num_feat_maps):
            sta = np.zeros((temp_size, size, size))
            for i, f in enumerate(files):
                data = np.load(folder + '/' + f)
                name, _ = os.path.splitext(f)
                base = os.path.basename(os.path.normpath(folder))
                sta += data[0, :, :, :, j]

            sta /= num_files
            spatial_kernel, temporal_kernel = filtertools.decompose_sta(sta)

            # Two subplots, the axes array is 1-d
            f, axarr = plt.subplots(2)
            axarr[0].set_title('STA Spatial Kernel')
            axarr[0].imshow(spatial_kernel)
            axarr[1].set_title('STA Temporal Kernel')
            axarr[1].plot(np.arange(-data.shape[1] + 1, 1), temporal_kernel)
            f.subplots_adjust(hspace=0.5)
            plt.savefig(output_folder + base + '-' + name + '_' + str(j) + '_' + '.png')
            plt.clf()
