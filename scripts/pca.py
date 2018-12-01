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
    layer_shapes = [(15, 56, 56, 192), (15, 28, 28, 192)]
    layer_names = ['Conv3d_2c_3x3', 'MaxPool3d_3a_3x3']
    layer_centers = [(27, 28), (13, 14)]

    # Output folder
    output_folder = 'output/' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '/'
    io_utils.create_folder(output_folder)

    # Info
    info = 'Functional caracterization of retinal ganglion cell diversity tool.'

    parser = argparse.ArgumentParser(description=info)
    parser.add_argument('--folder', metavar='folder', type=str, nargs='?', help='Simulation output folder')
    args = parser.parse_args()

    if args.folder is None:
        raise TypeError('ERROR: the arg --folder is mandatory [Simulation output folder]')

    # Logger
    log.create_logger(output_folder)
    logger = log.get_logger(__name__)
    logger.info(info)

    logger.info('Parameters:')
    logger.info('> folder: %s', args.folder)

    # Parameters
    kwargs = dict()
    kwargs['folder'] = args.folder
    kwargs['output_folder'] = output_folder
    kwargs['layer_shapes'] = layer_shapes
    kwargs['layer_names'] = layer_names
    kwargs['layer_centers'] = layer_centers

    # Compare correlated filters
    analyzer.analyze_principal_components_space_reduction(**kwargs)

    diff_time = time.time() - start_time
    logger.info('The simulation has been successfully completed.')

    logger.info('The process has been successfully completed.')
    logger.info('Elapsed time was {:.1f} seconds ({:.1f} minutes).'.format(diff_time, (diff_time / 60)))
