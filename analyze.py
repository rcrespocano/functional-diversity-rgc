# -*- coding: utf-8 -*-

import sys
import os
import time
import argparse
import datetime
from diversityrgc import log, analyzer, io_utils
from scripts import compare, decomposition, feature_maps_sta, pca


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
    number_of_spikes = 100
    layer_names = ['Conv3d_2c_3x3', 'MaxPool3d_3a_3x3']
    layer_shapes = [(15, 192), (15, 192)]
    layer_sizes = [56, 28]

    # Output folder
    output_folder = 'output/analyze-' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '/'
    io_utils.create_folder(output_folder)

    # Info
    info = 'Functional caracterization of retinal ganglion cell diversity tool: analyze RGC data.'

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

    # Analyze
    logger.info(' --> script: compare.py')
    compare.run(folder=args.folder)

    logger.info(' --> script: decomposition.py')
    decomposition.run(folder=args.folder)

    logger.info(' --> script: feature_maps_sta.py')
    feature_maps_sta.run(folder=args.folder)

    logger.info(' --> script: pca.py')
    pca.run(folder=args.folder)

    diff_time = time.time() - start_time
    logger.info('The simulation has been successfully completed.')

    logger.info('The process has been successfully completed.')
    logger.info('Elapsed time was {:.1f} seconds ({:.1f} minutes).'.format(diff_time, (diff_time / 60)))
