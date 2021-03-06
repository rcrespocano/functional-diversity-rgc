# -*- coding: utf-8 -*-

import sys
import os
import random
import time
import argparse
import datetime
from diversityrgc import log, analyzer, io_utils


if __name__ == '__main__':
    start_time = time.time()

    # Check Python version
    if sys.version_info[0] < 3:
        raise TypeError('This program must be executed with Python 3')

    # Do not show TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Parameters
    trigger_frames = 100
    frame_size = 56
    temporal_size = 30
    neural_net_image_size = 224
    number_of_spikes = None
    # Layers: Conv3d_2c_3x3 -> (1, 15, 56, 56, 192) and MaxPool3d_3a_3x3 -> (1, 15, 28, 28, 192)
    layer_names = ['Conv3d_2c_3x3', 'MaxPool3d_3a_3x3']
    layer_shapes = [(15, 192), (15, 192)]
    layer_sizes = [56, 28]

    # Output folder
    _dt = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    output_folder = 'output/run-' + _dt + '-' + str(random.randint(1,1000)) + '/'
    io_utils.create_folder(output_folder)

    # Info
    info = 'Functional caracterization of retinal ganglion cell diversity tool.'

    parser = argparse.ArgumentParser(description=info)
    parser.add_argument('--stimuli', metavar='stimuli', type=str, nargs='?', help='Input stimulus dataset')
    parser.add_argument('--bio', metavar='bio', type=str, nargs='?', help='Input biological data')
    parser.add_argument('--cells', metavar='cells', nargs='+', help='Cells (type,num)')
    args = parser.parse_args()

    if args.stimuli is None:
        raise TypeError('ERROR: the arg --stim is mandatory [Input stimulus dataset]')
    if args.bio is None:
        raise TypeError('ERROR: the arg --bio is mandatory [Input biological dat]')
    try:
        args.cells = [tuple(map(int, s.split(',', maxsplit=1))) for s in args.cells]
    except Exception:
        raise TypeError('ERROR: the arg --cells is mandatory [Cells (type,num)]')

    # Logger
    log.create_logger(output_folder)
    logger = log.get_logger(__name__)
    logger.info(info)

    logger.info('Parameters:')
    logger.info('> stim: %s', args.stimuli)
    logger.info('> bio: %s', args.bio)
    logger.info('> cells: %s', args.cells)

    # Parameters
    kwargs = dict()
    kwargs['stim_dataset'] = args.stimuli
    kwargs['bio_dataset'] = args.bio
    kwargs['cells'] = args.cells
    kwargs['trigger_frames'] = trigger_frames
    kwargs['frame_size'] = frame_size
    kwargs['temporal_size'] = temporal_size
    kwargs['neural_net_image_size'] = neural_net_image_size
    kwargs['output_folder'] = output_folder
    kwargs['number_of_spikes'] = number_of_spikes
    kwargs['layer_names'] = layer_names
    kwargs['layer_shapes'] = layer_shapes
    kwargs['layer_sizes'] = layer_sizes

    # Analyze
    analyzer.analyze(**kwargs)
    diff_time = time.time() - start_time
    logger.info('The simulation has been successfully completed.')

    logger.info('The process has been successfully completed.')
    logger.info('Elapsed time was {:.1f} seconds ({:.1f} minutes).'.format(diff_time, (diff_time / 60)))
