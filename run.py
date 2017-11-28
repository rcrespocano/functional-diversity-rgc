# -*- coding: utf-8 -*-

import sys
import os
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
    number_of_spikes = 10
    pcc = 0.5
    layer_name = 'MaxPool3d_3a_3x3'
    layer_shape = (8, 192)

    # Output folder
    output_folder = 'output/' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '/'
    io_utils.create_folder(output_folder)

    # Info
    info = 'Functional caracterization of retinal ganglion cell diversity tool.'

    parser = argparse.ArgumentParser(description=info)
    parser.add_argument('--stim', metavar='stim', type=str, nargs='?', help='Input stimulus dataset')
    parser.add_argument('--bio', metavar='bio', type=str, nargs='?', help='Input biological data')
    parser.add_argument('--cells', metavar='cells', nargs='+', help='Cells (type,num)')
    args = parser.parse_args()

    if args.stim is None:
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
    logger.info('> stim: %s', args.stim)
    logger.info('> bio: %s', args.bio)
    logger.info('> cells: %s', args.cells)

    # Parameters
    kwargs = dict()
    kwargs['stim_dataset'] = args.stim
    kwargs['bio_dataset'] = args.bio
    kwargs['cells'] = args.cells
    kwargs['trigger_frames'] = trigger_frames
    kwargs['frame_size'] = frame_size
    kwargs['temporal_size'] = temporal_size
    kwargs['neural_net_image_size'] = neural_net_image_size
    kwargs['output_folder'] = output_folder
    kwargs['number_of_spikes'] = number_of_spikes
    kwargs['pcc'] = pcc
    kwargs['layer_name'] = layer_name
    kwargs['layer_shape'] = layer_shape

    # Analyze
    analyzer.analyze(**kwargs)
    diff_time = time.time() - start_time
    logger.info('The simulation has been successfully completed.')

    logger.info('The process has been successfully completed.')
    logger.info('Elapsed time was {:.1f} seconds ({:.1f} minutes).'.format(diff_time, (diff_time / 60)))
