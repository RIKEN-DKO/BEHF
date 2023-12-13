#Based on https://github.com/autoliuweijie/BERT-whitening-pytorch

import numpy as np


def compute_kernel_bias(vecs):
    """ Compute Kernel and bias 
    y = (x + bias).dot(kernel)
    """
    # vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1/np.sqrt(s)))
    return W, -mu


def transform_and_normalize(vecs, kernel, bias):
    """apply the transform, then normalize
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return normalize(vecs)


def normalize(vecs):

    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
