# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from . import i3d
from . import log
from . import io_utils

# Logger
logger = log.get_logger(__name__)


class NeuralNet(object):
    def __init__(self, temporal_size, image_size, layer_names, layer_shapes, layer_sizes):
        self.temporal_size = temporal_size
        self.image_size = image_size
        self.layer_names = layer_names
        self.layer_shapes = layer_shapes
        self.layer_sizes = layer_sizes
        self.video_length = int(temporal_size)
        self.num_classes = 400
        self.label_map_path = 'data/label_map.txt'
        self.checkpoint_path = {'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt'}
        self.flags = tf.flags.FLAGS
        self.eval_type = 'rgb'
        tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
        tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
        tf.logging.set_verbosity(tf.logging.INFO)

        # RGB input has 3 channels.
        self.rgb_input = tf.placeholder(tf.float32, shape=(1, self.video_length, self.image_size, self.image_size, 3))
        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(self.num_classes, spatial_squeeze=True, final_endpoint=self.layer_names[-1])
            rgb_logits, all_endpoints = rgb_model(self.rgb_input, is_training=False, dropout_keep_prob=1.0)
            rgb_variable_map = {}
            for variable in tf.global_variables():
                if variable.name.split('/')[0] == 'RGB':
                    rgb_variable_map[variable.name.replace(':0', '')] = variable
            self.rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

        self.model_logits = rgb_logits
        self.model_endpoints = all_endpoints
        self.model_predictions = tf.nn.softmax(self.model_logits)

    def run(self, stimulus, sv, output_folder, nspikes=None, start=0, center_range=(55, 56), save=False, gif=False,
            gif_indexes=None, save_mean=False):
        with tf.Session() as sess:
            feed_dict = {}
            self.rgb_saver.restore(sess, self.checkpoint_path['rgb_imagenet'])
            logger.info('RGB checkpoint restored')

            # Format video file for the pre-trained network
            indexes_spike = np.nonzero(sv)[0]
            indexes_spike = indexes_spike[indexes_spike > self.temporal_size]

            # Number of spikes
            nspikes = len(indexes_spike) if nspikes is None else nspikes
            logger.info('Number of spikes: %s', nspikes)

            # Range
            indexes_spike = indexes_spike[start:start + nspikes]

            logger.info('Start simulation. Use each stimulus as input of the CNN.')
            cell_filters = [(np.zeros((nspikes,) + x)) for x in self.layer_shapes]

            # STAs
            stas = []
            for m in range(len(self.layer_shapes)):
                stas.append(np.zeros((1, self.layer_shapes[m][0], self.layer_sizes[m], self.layer_sizes[m], self.layer_shapes[m][1])))

            for i, index in enumerate(indexes_spike):
                # Current percentage
                io_utils.print_progress_bar(i, nspikes, prefix='Progress:', suffix='Complete', length=50)

                # Format video file
                video_file = np.zeros((self.video_length, self.image_size, self.image_size))
                for j, k in enumerate(range(index - (self.temporal_size - 1), index + 1)):
                    video_file[j] = io_utils.enlarge_image(stimulus[k], self.image_size, self.image_size)

                # Color dimension
                video_file = video_file[..., np.newaxis] * np.ones(3)

                # Reshape
                video_file = video_file.reshape((1,) + video_file.shape)

                # Predict
                feed_dict[self.rgb_input] = video_file
                (_) = sess.run([self.model_logits, self.model_predictions], feed_dict=feed_dict)

                # Activations
                for _idx, _layer in enumerate(self.layer_names):
                    filename = output_folder + _layer + '_' + (str(i).zfill(6))
                    units = sess.run(self.model_endpoints[_layer], feed_dict=feed_dict)

                    # Save sta
                    stas[_idx] += units

                    if save:
                        # Save all information
                        # units = units[0, :, center_range[0]:center_range[1]+1, center_range[0]:center_range[1]+1, :]
                        # units = np.mean(units, axis=(1, 2))
                        np.save(filename, units)

                        if save_mean:
                            cell_filters[_idx][i] = units

                    if gif:
                        if gif_indexes is None:
                            _iterator = range(units.shape[-1])
                        else:
                            _iterator = gif_indexes

                        for u in _iterator:
                                io_utils.generate_gif(filename + '_' + str(u) + '.gif', units[0, :, :, :, u], fps=30)

            if save and save_mean:
                for k, feature_map in enumerate(cell_filters):
                    fm = np.mean(feature_map, axis=0)
                    np.save(output_folder + 'mean_' + str(k), fm)

            # STAs
            stas = [sta/len(indexes_spike) for sta in stas]
            for l, sta in enumerate(stas):
                _sta = sta / len(indexes_spike)
                np.save(output_folder + self.layer_names[l] + '_sta', _sta)

    def run_default_stim(self, gif_indexes, output_folder):
        # Load video file
        video_file = np.load('data/v_CricketShot_g04_c01_rgb.npy')
        video_file = video_file[:, :self.video_length, :, :, :]

        with tf.Session() as sess:
            feed_dict = {}
            self.rgb_saver.restore(sess, self.checkpoint_path['rgb_imagenet'])

            # Predict
            feed_dict[self.rgb_input] = video_file
            (_) = sess.run([self.model_logits, self.model_predictions], feed_dict=feed_dict)

            # Activations
            filename = output_folder + self.layer_name + '_DEFAULT'
            units = sess.run(self.model_endpoints[self.layer_name], feed_dict=feed_dict)

            for u in gif_indexes:
                io_utils.generate_gif(filename + '_' + str(u) + '.gif', units[0, :, :, :, u], fps=30)
