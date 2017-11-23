# -*- coding: utf-8 -*-

import sys
import os
import time
import datetime
from diversityrgc import log, analyzer


if __name__ == '__main__':
    start_time = time.time()

    # Check Python version
    if sys.version_info[0] < 3:
        raise TypeError('This program must be executed with Python 3')

    # Parameters
    stim_dataset = '/home/rcc/Recordings/stim-pink-noise.mat'
    bio_dataset = '/home/rcc/Recordings/20050825_serial_003.mat'
    cell_type_index = 7
    cell_number = 7
    trigger_frames = 100
    frame_size = 56
    temporal_size = 20
    neural_net_image_size = 224

    # Output folder
    output_folder = 'output/' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '/'
    os.makedirs(output_folder)

    # Info
    info = 'Functional caracterization of retinal ganglion cell diversity tool.'

    # Logger
    log.create_logger(output_folder)
    logger = log.get_logger(__name__)
    logger.info(info)

    # Parameters
    kwargs = dict()
    kwargs['stim_dataset'] = stim_dataset
    kwargs['bio_dataset'] = bio_dataset
    kwargs['cell_type_index'] = cell_type_index
    kwargs['cell_number'] = cell_number
    kwargs['trigger_frames'] = trigger_frames
    kwargs['frame_size'] = frame_size
    kwargs['temporal_size'] = temporal_size
    kwargs['neural_net_image_size'] = neural_net_image_size
    kwargs['output_folder'] = output_folder

    # Analyze
    analyzer.analyze(**kwargs)
    diff_time = time.time() - start_time
    logger.info('The simulation has been successfully completed.')

    logger.info('The process has been successfully completed.')
    logger.info('Elapsed time was {:.1f} seconds ({:.1f} minutes).'.format(diff_time, (diff_time / 60)))
