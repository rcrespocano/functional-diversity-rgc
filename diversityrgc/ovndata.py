# -*- coding: utf-8 -*-

import h5py
import numpy as np
import matplotlib.pyplot as plt
from . import log


# Constants
DATASET_KEY = 'stimpinknoise'
BIODATA_KEY = 'datarun'
CELL_TYPES_KEY = 'cell_types'
CELL_TYPE_NAME_KEY = 'name'
CELL_TYPE_CELL_IDS_KEY = 'cell_ids'
TRIGGERS_KEY = 'triggers'
SPIKES_KEY = 'spikes'
VISION_KEY = 'vision'
STA_FITS_KEY = 'sta_fits'
MEAN_KEY = 'mean'


# Logger
logger = log.get_logger(__name__)


def load_data(**kwargs):
    stim_dataset = kwargs['stim_dataset']
    bio_dataset = kwargs['bio_dataset']
    cell_type_index = kwargs['cell_type_index']
    cell_number = kwargs['cell_number']
    trigger_frames = kwargs['trigger_frames']
    frame_size = kwargs['frame_size']
    temporal_size = kwargs['temporal_size']
    output_folder = kwargs['output_folder']
    subframe_size = int(frame_size / 2)

    # Stimulus dataset
    logger.info('Load stimulus dataset.')
    with h5py.File(stim_dataset) as f:
        stim_dataset_frames = f[DATASET_KEY][:]
        logger.info('Stimulus dataset raw (shape): %s', stim_dataset_frames.shape)

    # Get labels/spikes/triggers of biodataset
    logger.info('Load bio dataset.')
    with h5py.File(bio_dataset) as f:
        # Cell type
        ct_ref = f[BIODATA_KEY][CELL_TYPES_KEY][cell_type_index - 1][0]
        cell_type = f[ct_ref]
        name = u''.join(chr(c) for c in cell_type[CELL_TYPE_NAME_KEY].value)
        cell_ids = cell_type[CELL_TYPE_CELL_IDS_KEY]

        # Spikes
        idx = cell_ids[cell_number - 1][0]
        cell_ids = (np.array(f[BIODATA_KEY][CELL_TYPE_CELL_IDS_KEY]).flatten()).astype(int)
        index = np.where(cell_ids == idx)[0][0]
        spikes_ref = f[BIODATA_KEY][SPIKES_KEY][0][index]
        data_spikes = f[spikes_ref][0]

        # Triggers
        data_triggers = f[BIODATA_KEY][TRIGGERS_KEY][0]

        # STA fits (mean)
        fits_ref = f[BIODATA_KEY][VISION_KEY][STA_FITS_KEY][0][index]
        fits = f[fits_ref][MEAN_KEY]
        mean = np.array(fits.value).flatten()
        mean[0], mean[1] = mean[1], mean[0]

        logger.info('Cell: %s', idx)
        logger.info('Cell Type name: %s', name)
        logger.info('Cell Type index: %s', cell_type_index)
        logger.info('Cell Type number: %s', cell_number)
        logger.info('Data Triggers (shape): %s', data_triggers.shape)
        logger.info('Data Spikes (shape): %s', data_spikes.shape)
        logger.info('STA fits (mean): %s', mean)

    # Divide triggers for each block of frames
    triggers = np.zeros(((len(data_triggers) - 1) * trigger_frames) + 1)
    triggers[-1] = data_triggers[-1]
    logger.info('Triggers (shape): %s', triggers.shape)

    for i in range(0, len(data_triggers) - 1):
        _bin = (data_triggers[i + 1] - data_triggers[i]) / trigger_frames
        triggers[i * trigger_frames] = data_triggers[i]

        for j in range(1, trigger_frames):
            triggers[(i * trigger_frames) + j] = data_triggers[i] + (j * _bin)

    # Spike vector (sv)
    sv = np.zeros(len(triggers) - 1)
    for i in range(0, len(triggers) - 1):
        cond = np.logical_and(data_spikes >= triggers[i], data_spikes < triggers[i + 1])
        n = len(data_spikes[cond])
        sv[i] = 1 if (n > 0) else 0

    # Stimulus dataset (reduce to the size of the spike vector)
    stim_dataset_frames = stim_dataset_frames[:sv.shape[0]]

    # Spike vector size
    spike_vector_spikes_size = np.count_nonzero(sv)

    logger.info('Spike Vector (shape): %s', sv.shape)
    logger.info('Spike Vector spikes: %s', spike_vector_spikes_size)
    logger.info('Spike Vector non spikes: %s', np.where(sv == 0)[0].size)

    # Crop frames (reduce to the receptive field zone of the cell)
    logger.info('Reduce frame to the receptive field zone of the cell')
    dataset_cropped_frames = np.zeros((stim_dataset_frames.shape[0], frame_size, frame_size))
    for i in range(dataset_cropped_frames.shape[0]):
        dataset_cropped_frames[i] = crop_frame(stim_dataset_frames[i], mean, subframe_size)

    # Normalize
    logger.info('Normalize stimulus')
    dataset_cropped_frames = (2 * dataset_cropped_frames.astype('float32') / 255.0) -1

    # Save STA
    logger.info('Calculate and save STA')
    save_sta(sv, dataset_cropped_frames, size=temporal_size, output_folder=output_folder)

    return dataset_cropped_frames, sv


def crop_frame(frame, mean, subframe_size):
    side_length = subframe_size * 2
    max_y, max_x = frame.shape
    x = mean.astype(int)[0]
    y = mean.astype(int)[1]

    # Get indexes range (start and stop) of x and y axes
    start_x = 0 if (x - subframe_size < 0) else x - subframe_size
    stop_x = max_x if (x + subframe_size > max_x) else x + subframe_size
    start_y = 0 if (y - subframe_size < 0) else y - subframe_size
    stop_y = max_y if (y + subframe_size > max_y) else y + subframe_size

    # Adjust to obtain a squared frame
    if (stop_x - start_x) < side_length:
        if start_x == 0:
            stop_x += (side_length - (stop_x - start_x))
        else:
            start_x -= (side_length - (stop_x - start_x))

    if (stop_y - start_y) < (subframe_size * 2):
        if start_y == 0:
            stop_y += (side_length - (stop_y - start_y))
        else:
            start_y -= (side_length - (stop_y - start_y))

    return frame[start_y:stop_y, start_x:stop_x]


def save_sta(spike_vector, stimulus, size, output_folder, name='sta'):
    sta_plot_cols = 10
    sta_plot_rows = (size // sta_plot_cols) + 1

    # Get indexes of spikes in spike vector
    indexes = np.nonzero(spike_vector)[0]

    # Calculate STA
    plot_frames = []

    for i in range(size - 1, -1, -1):
        frame = np.zeros(stimulus[0].shape)
        for index in indexes:
            frame = np.add(frame, stimulus[index - i])

        # Calculate the avg of the frames
        frame /= len(indexes)

        # Save frame for plotting
        plot_frames.append(frame)

    # STA plot
    f, axarr = plt.subplots(sta_plot_rows, sta_plot_cols)
    for i in range(sta_plot_rows):
        for j in range(sta_plot_cols):
            axarr[i, j].axis('off')

    for i in range(len(plot_frames)):
        x = i // sta_plot_cols
        y = i % sta_plot_cols
        axarr[x, y].imshow(plot_frames[i], cmap='gray')

    plt.savefig(output_folder + name + '.pdf')
    plt.clf()
