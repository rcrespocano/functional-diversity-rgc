# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from . import i3d
from . import log
from . import io_utils

# Logger
logger = log.get_logger(__name__)


class NeuralNet(object):
    def __init__(self, temporal_size, image_size, output_folder):
        self.temporal_size = temporal_size
        self.output_folder = output_folder
        self.image_size = image_size
        self.video_length = int(temporal_size / 2)
        self.num_classes = 400
        self.label_map_path = 'data/label_map.txt'
        self.checkpoint_path = {'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt'}
        self.flags = tf.flags.FLAGS
        self.eval_type = 'rgb'

        tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
        tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
        tf.logging.set_verbosity(tf.logging.INFO)
        self.imagenet_pretrained = self.flags.imagenet_pretrained

        # RGB input has 3 channels.
        self.rgb_input = tf.placeholder(tf.float32, shape=(1, self.video_length, self.image_size, self.image_size, 3))
        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(self.num_classes, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, all_endpoints = rgb_model(self.rgb_input, is_training=False, dropout_keep_prob=1.0)
            rgb_variable_map = {}
            for variable in tf.global_variables():
                if variable.name.split('/')[0] == 'RGB':
                    rgb_variable_map[variable.name.replace(':0', '')] = variable
            self.rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

        self.model_logits = rgb_logits
        self.model_endpoints = all_endpoints
        self.model_predictions = tf.nn.softmax(self.model_logits)

    def run(self, stimulus, sv, nspikes=None, save=False, gif=False, gif_indexes=None):
        with tf.Session() as sess:
            feed_dict = {}
            self.rgb_saver.restore(sess, self.checkpoint_path['rgb_imagenet'])
            logger.info('RGB checkpoint restored')

            # Format video file for the pre-trained network
            indexes_spike = np.nonzero(sv)[0]
            number_of_spikes = len(indexes_spike)
            nspikes = number_of_spikes if nspikes is None else nspikes

            logger.info('Start simulation. Use each stimulus as input of the CNN.')
            for i, index in enumerate(indexes_spike):
                # Current percentage
                io_utils.print_progress_bar(i, nspikes, prefix='Progress:', suffix='Complete', length=50)

                # Format video file
                video_file = np.zeros((self.video_length, self.image_size, self.image_size))
                for j, k in enumerate(range(index - (self.temporal_size - 1), index + 1, 2)):
                    image_one = io_utils.enlarge_image(stimulus[k], self.image_size, self.image_size)
                    image_two = io_utils.enlarge_image(stimulus[k+1], self.image_size, self.image_size)
                    video_file[j] = np.mean(np.array([image_one, image_two]), axis=0)

                # Color dimension
                video_file = video_file[..., np.newaxis] * np.ones(3)

                # Reshape
                video_file = video_file.reshape((1,) + video_file.shape)

                # Predict
                feed_dict[self.rgb_input] = video_file
                (_) = sess.run([self.model_logits, self.model_predictions], feed_dict=feed_dict)

                # Activations
                layer = 'MaxPool3d_3a_3x3'
                filename = self.output_folder + layer + '_' + (str(i).zfill(6))
                units = sess.run(self.model_endpoints[layer], feed_dict=feed_dict)

                if save:
                    units = units[0, :, :, :, :]
                    np.save(filename, units)

                if gif:
                    if gif_indexes is None:
                        _iterator = range(units.shape[-1])
                    else:
                        _iterator = gif_indexes

                    for u in _iterator:
                            io_utils.generate_gif(filename + '_' + str(u) + '.gif', units[0, :, :, :, u], fps=30)

                if (i+1) >= nspikes:
                    break
