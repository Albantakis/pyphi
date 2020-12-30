#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# metrics/distribution.py

"""Metrics on probability distributions."""

from contextlib import ContextDecorator
from math import log2

import numpy as np
from pyemd import emd as _emd
from scipy.spatial.distance import cdist
from scipy.special import entr, rel_entr

from .. import config, constants, utils, validate
from ..direction import Direction
from ..distribution import flatten, marginal_zero
from ..registry import Registry

_LN_OF_2 = np.log(2)


class DistributionMeasureRegistry(Registry):
    """Storage for distance functions between probability distributions.

    Users can define custom measures:

    Examples:
        >>> @measures.register('ALWAYS_ZERO')  # doctest: +SKIP
        ... def always_zero(a, b):
        ...    return 0

    And use them by setting, *e.g.*, ``config.REPERTOIRE_DISTANCE = 'ALWAYS_ZERO'``.
    """

    # pylint: disable=arguments-differ

    desc = "distance functions between probability distributions"

    def __init__(self):
        super().__init__()
        self._asymmetric = []

    def register(self, name, asymmetric=False):
        """Decorator for registering a distribution measure with PyPhi.

        Args:
            name (string): The name of the measure.

        Keyword Args:
            asymmetric (boolean): ``True`` if the measure is asymmetric.
        """

        def register_func(func):
            if asymmetric:
                self._asymmetric.append(name)
            self.store[name] = func
            return func

        return register_func

    def asymmetric(self):
        """Return a list of asymmetric measures."""
        return self._asymmetric


measures = DistributionMeasureRegistry()


class np_suppress(np.errstate, ContextDecorator):
    """Decorator to suppress NumPy warnings about divide-by-zero and
    multiplication of ``NaN``.

    .. note::
        This should only be used in cases where you are *sure* that these
        warnings are not indicative of deeper issues in your code.
    """

    def __init__(self):
        super().__init__(divide="ignore", invalid="ignore")


# Load precomputed hamming matrices.
_NUM_PRECOMPUTED_HAMMING_MATRICES = 10
_hamming_matrices = utils.load_data(
    "hamming_matrices", _NUM_PRECOMPUTED_HAMMING_MATRICES
)


# TODO extend to nonbinary nodes
def _hamming_matrix(N):
    """Return a matrix of Hamming distances for the possible states of |N|
    binary nodes.

    Args:
        N (int): The number of nodes under consideration

    Returns:
        np.ndarray: A |2^N x 2^N| matrix where the |ith| element is the Hamming
        distance between state |i| and state |j|.

    Example:
        >>> _hamming_matrix(2)
        array([[0., 1., 1., 2.],
               [1., 0., 2., 1.],
               [1., 2., 0., 1.],
               [2., 1., 1., 0.]])
    """
    if N < _NUM_PRECOMPUTED_HAMMING_MATRICES:
        return _hamming_matrices[N]
    return _compute_hamming_matrix(N)


@constants.joblib_memory.cache
def _compute_hamming_matrix(N):
    """Compute and store a Hamming matrix for |N| nodes.

    Hamming matrices have the following sizes::

        N   MBs
        ==  ===
        9   2
        10  8
        11  32
        12  128
        13  512

    Given these sizes and the fact that large matrices are needed infrequently,
    we store computed matrices using the Joblib filesystem cache instead of
    adding computed matrices to the ``_hamming_matrices`` global and clogging
    up memory.

    This function is only called when |N| >
    ``_NUM_PRECOMPUTED_HAMMING_MATRICES``. Don't call this function directly;
    use |_hamming_matrix| instead.
    """
    possible_states = np.array(list(utils.all_states((N))))
    return cdist(possible_states, possible_states, "hamming") * N


# TODO extend to nonbinary nodes
def hamming_emd(p, q):
    """Return the Earth Mover's Distance between two distributions (indexed
    by state, one dimension per node) using the Hamming distance between states
    as the transportation cost function.

    Singleton dimensions are sqeezed out.
    """
    N = p.squeeze().ndim
    p, q = flatten(p), flatten(q)
    return _emd(p, q, _hamming_matrix(N))


def effect_emd(p, q):
    """Compute the EMD between two effect repertoires.

    Because the nodes are independent, the EMD between effect repertoires is
    equal to the sum of the EMDs between the marginal distributions of each
    node, and the EMD between marginal distribution for a node is the absolute
    difference in the probabilities that the node is OFF.

    Args:
        p (np.ndarray): The first repertoire.
        q (np.ndarray): The second repertoire.

    Returns:
        float: The EMD between ``p`` and ``q``.
    """
    return sum(abs(marginal_zero(p, i) - marginal_zero(q, i)) for i in range(p.ndim))


@measures.register("EMD")
def emd(p, q, direction):
    """Compute the EMD between two repertoires for a given direction.

    The full EMD computation is used for cause repertoires. A fast analytic
    solution is used for effect repertoires.

    Args:
        p (np.ndarray): The first repertoire.
        q (np.ndarray): The second repertoire.
        direction (Direction): |CAUSE| or |EFFECT|.

    Returns:
        float: The EMD between ``p`` and ``q``, rounded to |PRECISION|.

    Raises:
        ValueError: If ``direction`` is invalid.
    """
    if (direction == Direction.CAUSE) or (direction is None):
        func = hamming_emd
    elif direction == Direction.EFFECT:
        func = effect_emd
    else:
        # TODO: test that ValueError is raised
        validate.direction(direction)

    return round(func(p, q), config.PRECISION)


@measures.register("L1")
def l1(p, q):
    """Return the L1 distance between two distributions.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The sum of absolute differences of ``p`` and ``q``.
    """
    return np.abs(p - q).sum()


@measures.register("ENTROPY_DIFFERENCE")
def entropy_difference(p, q):
    """Return the difference in entropy between two distributions."""
    hp = entr(p).sum() / _LN_OF_2
    hq = entr(q).sum() / _LN_OF_2
    return abs(hp - hq)


@measures.register("PSQ2")
def psq2(p, q):
    r"""Compute the PSQ2 measure.

    This is defined as :math:`\mid f(p) - f(q) \mid`, where

    .. math::
        f(x) = \sum_{i=0}^{N-1} p_i^2 \log_2 (p_i N)

    Args:
        p (np.ndarray): The first distribution.
        q (np.ndarray): The second distribution.
    """
    fp = (p * (-1.0 * entr(p))).sum() / _LN_OF_2 + (p ** 2 * log2(len(p))).sum()
    fq = (q * (-1.0 * entr(q))).sum() / _LN_OF_2 + (q ** 2 * log2(len(q))).sum()
    return abs(fp - fq)


@measures.register("MP2Q", asymmetric=True)
@np_suppress()
def mp2q(p, q):
    r"""Compute the MP2Q measure.

    This is defined as

    .. math::
        \frac{1}{N}
        \sum_{i=0}^{N-1} \frac{p_i^2}{q_i} \log_2\left(\frac{p_i}{q_i}\right)

    Args:
        p (np.ndarray): The first distribution.
        q (np.ndarray): The second distribution.

    Returns:
        float: The distance.
    """
    return np.sum(p / q * information_density(p, q) / len(p))


def information_density(p, q):
    """Return the information density of p relative to q, in base 2.

    This is also known as the element-wise relative entropy; see
    :func:`scipy.special.rel_entr`.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        np.ndarray: The information density of ``p`` relative to ``q``.
    """
    return rel_entr(p, q) / _LN_OF_2


@measures.register("KLD", asymmetric=True)
def kld(p, q):
    """Return the Kullback-Leibler Divergence (KLD) between two distributions.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The KLD of ``p`` from ``q``.
    """
    return information_density(p, q).sum()


def absolute_information_density(p, q):
    """Return the absolute information density function of two distributions.

    The information density is also known as the element-wise relative
    entropy; see :func:`scipy.special.rel_entr`.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        np.ndarray: The absolute information density of ``p`` relative to ``q``.
    """
    return np.abs(information_density(p, q))


def maximal_state(repertoire, partitioned_repertoire):
    """Return the state(s) with the maximal AID between the repertoires.

    Note that there can be ties.

    Returns:
        np.ndarray: A 2D array where each row is a maximal state.
    """
    # TODO(4.0) this is unnecessarily recomputed; should make a
    # DistanceResult class that can carry auxilliary data, e.g. the maximal
    # states
    density = absolute_information_density(
        repertoire.squeeze(), partitioned_repertoire.squeeze()
    )
    return tuple(np.transpose(np.where(density == density.max())).flat)


@measures.register("ID", asymmetric=True)
def intrinsic_difference(p, q):
    r"""Compute the intrinsic difference (ID) between two distributions.

    This is defined as

    .. math::
        \max_i \left\{
            p_i \log_2 \left( \frac{p_i}{q_i} \right)
        \right\}

    where we define :math:`p_i \log_2 \left( \frac{p_i}{q_i} \right)` to be
    :math:`0` when :math:`p_i = 0` or :math:`q_i = 0`.

    See the following paper:

        Barbosa LS, Marshall W, Streipert S, Albantakis L, Tononi G (2020).
        A measure for intrinsic information.
        *Sci Rep*, 10, 18803. https://doi.org/10.1038/s41598-020-75943-4

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The intrinsic difference.
    """
    return np.max(information_density(p, q))


@measures.register("AID", asymmetric=True)
@measures.register("KLM", asymmetric=True)  # Backwards-compatible alias
@measures.register("BLD", asymmetric=True)  # Backwards-compatible alias
def absolute_intrinsic_difference(p, q):
    """Compute the absolute intrinsic difference (AID) between two
    distributions.

    This is the same as the ID, but with the absolute value taken before the
    maximum is taken.

    See documentation for :func:`intrinsic_difference` for further details
    and references.

    Args:
        p (float): The first probability distribution.
        q (float): The second probability distribution.

    Returns:
        float: The absolute intrinsic difference.
    """
    return np.max(absolute_information_density(p, q))
