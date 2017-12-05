# -*- coding: utf-8 -*-
"""
    Tools and utilities for basic filter signal processing.

    Original GitHub repo: https://github.com/baccuslab/pyret/blob/master/pyret/filtertools.py
"""

import numpy as np


def sta(spike_vector, stimulus, n_samples):
    # Get indexes of spikes in spike vector
    indexes = np.nonzero(spike_vector)[0]

    # Initialize STA
    sta = np.zeros((n_samples,) + stimulus[0].shape)

    # Calculate STA
    for i in range(n_samples - 1, -1, -1):
        sta_index = abs(i-n_samples+1)
        sta[sta_index] = np.zeros(stimulus[0].shape)
        for index in indexes:
            sta[sta_index] = np.add(sta[sta_index], stimulus[index - i])

        # Calculate the avg of the frames
        sta[sta_index] /= len(indexes)

    return sta


def sta_dataset(spike_vector, dataset):
    if len(dataset.shape) != 4:
        dataset = dataset.reshape(dataset.shape[0:4])

    n_samples = dataset.shape[1]

    # Get indexes of spikes in spike vector
    indexes = np.nonzero(spike_vector)[0]

    # Initialize STA
    sta = np.zeros(dataset.shape[1:4])

    # Calculate STA
    for i in range(n_samples - 1, -1, -1):
        sta_index = abs(i - n_samples + 1)
        sta[sta_index] = np.zeros(dataset.shape[2:4])
        for index in indexes:
            sta[sta_index] = np.add(sta[sta_index], dataset[index - i][sta_index])

        # Calculate the avg of the frames
        sta[sta_index] /= len(indexes)

    return sta


def decompose_sta(sta):
    """
    Decomposes a spatiotemporal STA into a spatial and temporal kernel.
    Parameters
    ----------
    sta : array_like
        The full 3-dimensional STA to be decomposed, of shape ``(t, nx, ny)``.
    Returns
    -------
    s : array_like
        The spatial kernel, with shape ``(nx * ny,)``.
    t : array_like
        The temporal kernel, with shape ``(t,)``.
    """
    _, u, _, v = low_rank_sta(sta, k=1)
    return v[0].reshape(sta.shape[1:]), u[:, 0]


def low_rank_sta(sta_orig, k=10):
    """
    Constructs a rank-k approximation to the given spatiotemporal STA.
    This is useful for estimating a spatial and temporal kernel for an
    STA or for denoising.
    Parameters
    ----------
    sta_orig : array_like
        3D STA to be separated, shaped as ``(time, space, space)``.
    k : int
        Number of components to keep (rank of the reduced STA).
    Returns
    -------
    sk : array_like
        The rank-k estimate of the original STA.
    u : array_like
        The top ``k`` temporal components (each column is a component).
    s : array_like
        The top ``k`` singular values.
    v : array_like
        The top ``k`` spatial components (each row is a component). These
        components have all spatial dimensions collapsed to one.
    Notes
    -----
    This method requires that the STA be 3D. To decompose a STA into a
    temporal and 1-dimensional spatial component, simply promote the STA
    to 3D before calling this method.
    Despite the name this method accepts both an STA or a linear filter.
    The components estimated for one will be flipped versions of the other.
    """

    # work with a copy of the STA (prevents corrupting the input)
    f = sta_orig.copy() - sta_orig.mean()

    # Compute the SVD of the full STA
    assert f.ndim >= 2, 'STA must be at least 2-D'
    u, s, v = np.linalg.svd(f.reshape(f.shape[0], -1), full_matrices=False)

    # Keep the top k components
    k = np.min([k, s.size])
    u = u[:, :k]
    s = s[:k]
    v = v[:k, :]

    # Compute the rank-k STA
    sk = (u.dot(np.diag(s).dot(v))).reshape(f.shape)

    # Ensure that the computed STA components have the correct sign.
    # The full STA should have positive projection onto first temporal
    # component of the low-rank STA.
    sign = np.sign(np.einsum('i,ijk->jk', u[:, 0], f).sum())
    u *= sign
    v *= sign

    # Return the rank-k approximate STA, and the SVD components
    return sk, u, s, v
