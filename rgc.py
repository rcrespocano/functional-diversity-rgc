# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import i3d
import imageio
import time
import datetime
import os

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


if __name__ == '__main__':
    # Files
    stim_dataset = '/home/rcc/Research/data/stim-pink-noise.mat'
    bio_dataset = '/home/rcc/Research/data/20050825_serial_003.mat'

    # Parameters
    cell_type_index = 7
    cell_number = 7
    trigger_frames = 100
    frame_size = 56
    temporal_size = 20
    pretrained_network_image_size = 224

    subframe_size = int(frame_size / 2)

    # Start
    start_time = time.time()

    # Output folder
    output_folder = '/tmp/' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '/'
    os.makedirs(output_folder)

    # Stimulus dataset
    print('Load stimulus dataset.')
    with h5py.File(stim_dataset) as f:
        stim_dataset_frames = f[DATASET_KEY][:]
        print(' > Stimulus dataset raw (shape):', stim_dataset_frames.shape)

    # Get labels/spikes/triggers of biodataset
    print('Load bio dataset.')
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

        print(' > Cell:', idx)
        print(' > Cell Type name:', name)
        print(' > Cell Type index:', cell_type_index)
        print(' > Cell Type number:', cell_number)
        print(' > Data Triggers (shape):', data_triggers.shape)
        print(' > Data Spikes (shape):', data_spikes.shape)
        print(' > STA fits (mean):', mean)

    # Divide triggers for each block of frames
    triggers = np.zeros(((len(data_triggers) - 1) * trigger_frames) + 1)
    triggers[-1] = data_triggers[-1]
    print(' > Triggers (shape):', triggers.shape)

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

    print(' > Spike Vector (shape):', sv.shape)
    print(' > Spike Vector spikes:', spike_vector_spikes_size)
    print(' > Spike Vector non spikes:', np.where(sv == 0)[0].size)

    # Crop frames (reduce to the receptive field zone of the cell)
    print(' > Reduce frame to the receptive field zone of the cell')
    dataset_cropped_frames = np.zeros((stim_dataset_frames.shape[0], frame_size, frame_size))
    for i in range(dataset_cropped_frames.shape[0]):
        dataset_cropped_frames[i] = crop_frame(stim_dataset_frames[i], mean, subframe_size)

    # Normalize
    print(' > Normalize stimulus')
    dataset_cropped_frames = (2 * dataset_cropped_frames.astype('float32') / 255.0) -1

    # Save STA
    print(' > Calculate and save STA')
    save_sta(sv, dataset_cropped_frames, size=temporal_size, output_folder=output_folder)

    # Classify video file using a trained Kinetics checkpoint
    print('Classify video file using a trained Kinetics checkpoint.')

    _NUM_CLASSES = 400
    _LABEL_MAP_PATH = 'data/label_map.txt'

    _CHECKPOINT_PATHS = {
        'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
        'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    }

    FLAGS = tf.flags.FLAGS

    tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
    tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
    tf.logging.set_verbosity(tf.logging.INFO)
    imagenet_pretrained = FLAGS.imagenet_pretrained
    eval_type = 'rgb'
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]
    print(' > Eval type', eval_type)

    # RGB input has 3 channels.
    rgb_input = tf.placeholder(tf.float32, shape=(1, temporal_size, pretrained_network_image_size, pretrained_network_image_size, 3))
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
        rgb_logits, all_endpoints = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)


    model_logits = rgb_logits
    model_predictions = tf.nn.softmax(model_logits)

    # Format video file for the pre-trained network
    indexes_spike = np.nonzero(sv)[0]
    for i, index in enumerate(indexes_spike):
        print('Processing spike %5d of %5d' % (i, len(indexes_spike)))

        video_file = np.zeros((temporal_size, 224, 224))
        for j, k in enumerate(range(index - (temporal_size - 1), index + 1)):
            video_file[j,:frame_size,:frame_size] = dataset_cropped_frames[k]

        # Color dimension
        video_file = video_file[..., np.newaxis] * np.ones(3)

        # Reshape
        video_file = video_file.reshape((1,) + video_file.shape)       
        
        with tf.Session() as sess:
            feed_dict = {}
              
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
            
            tf.logging.info('RGB checkpoint restored')
            rgb_sample = video_file
            tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
            feed_dict[rgb_input] = rgb_sample

            # Predict
            out_logits, out_predictions = sess.run([model_logits, model_predictions], feed_dict=feed_dict)

            # Activations
            layers = ['MaxPool3d_3a_3x3']

            for layer in layers:
                units = sess.run(all_endpoints[layer], feed_dict=feed_dict)
                print('Layer ' + layer + '. Shape:', units[0,:,:,:,0].shape)
                for i in range(units.shape[-1]):
                    imageio.mimwrite(output_folder + layer + str(i) + '.gif', units[0,:,:,:,i], fps=30)

    diff_time = (time.time() - start_time) / 60
    print('Elapsed time in minutes:', diff_time)

