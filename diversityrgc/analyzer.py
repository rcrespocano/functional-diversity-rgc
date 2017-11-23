# -*- coding: utf-8 -*-

from . import ovndata
from . import neuralnet


def analyze(**kwargs):
    # Load data
    stim, sv = ovndata.load_data(**kwargs)

    # Convolutional neural network model for video classification trained on the Kinetics dataset
    net = neuralnet.NeuralNet(temporal_size=kwargs['temporal_size'],
                              image_size=kwargs['neural_net_image_size'],
                              output_folder=kwargs['output_folder'])
    net.run(stim, sv, nspikes=10)
