# Copyright 2019 Matthew J. Aburn
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See <http://www.gnu.org/licenses/>.

"""Numerical integration algorithms for fractional ordinary differential
   equations.

Usage:
    fodeint(a, f, y0, tspan)  for Caputo equation D^a y(t) = f(y,t)

    a is the fractional exponent (currently must have 0<a<1)
    y0 is the initial value
    tspan is an array of time values (currently these must be equally spaced)
    function f is the right hand side of the system (scalar or  dx1  vector)

    caputoEuler(a, f, y0, tspan): the explicit one-step Adams-Bashforth (Euler)
      method. The simplest possible method.
"""

from __future__ import absolute_import
import numpy as np
import numbers
from scipy import special


class Error(Exception):
    pass


class FODEValueError(Error):
    """Thrown if integration arguments fail some basic sanity checks"""
    pass


def _check_args(a, f, y0, tspan):
    """Do some validation common to all algorithms. Find dimension d."""
    if not (a > 0.0 and a < 1.0):
        raise FODEValueError('Currently `a` must be in the range (0,1).')
    if not np.isclose(min(np.diff(tspan)), max(np.diff(tspan))):
        raise FODEValueError('Currently time steps must be equally spaced.')
    # Be flexible to allow scalar equations. convert them to a 1D vector system
    if isinstance(y0, numbers.Number):
        if isinstance(y0, numbers.Integral):
            numtype = np.float64
        else:
            numtype = type(y0)
        y0_orig = y0
        y0 = np.array([y0], dtype=numtype)
        def make_vector_fn(fn):
            def newfn(y, t):
                return np.array([fn(y[0], t)], dtype=numtype)
            newfn.__name__ = fn.__name__
            return newfn
        if isinstance(f(y0_orig, tspan[0]), numbers.Number):
            f = make_vector_fn(f)
    # determine dimension d of the system
    d = len(y0)
    if len(f(y0, tspan[0])) != d:
        raise FODEValueError('y0 and f have incompatible shapes.')
    return (d, a, f, y0, tspan)


def fodeint(a, f, y0, tspan):
    """ Numerically integrate the Caputo equation D^a y(t) = f(y,t)

    where a is the fractional exponent of the Caputo differential operator,
    y is the d-dimensional state vector, and f is a vector-valued function.

    Args:
      a: fractional exponent in the range (0,1)
      f: callable(y,t) returning a numpy array of shape (d,)
         Vector-valued function to define the right hand side of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.

    Returns:
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row

    Raises:
      FODEValueError
    """
    # In a future version we can automatically choose here the most suitable
    # algorithm based on properties of the system.
    (d, a, f, y0, tspan) = _check_args(a, f, y0, tspan)
    chosenAlgorithm = caputoEuler
    return chosenAlgorithm(a, f, y0, tspan)


def caputoEuler(a, f, y0, tspan):
    """Use one-step Adams-Bashforth (Euler) method to integrate Caputo equation
    D^a y(t) = f(y,t)

    Args:
      a: fractional exponent in the range (0,1)
      f: callable(y,t) returning a numpy array of shape (d,)
         Vector-valued function to define the right hand side of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.

    Returns:
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row

    Raises:
      FODEValueError

    See also:
      K. Diethelm et al. (2004) Detailed error analysis for a fractional Adams
         method
      C. Li and F. Zeng (2012) Finite Difference Methods for Fractional
         Differential Equations
    """
    (d, a, f, y0, tspan) = _check_args(a, f, y0, tspan)
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1)
    c = special.rgamma(a) * np.power(h, a) / a
    w = c * np.diff(np.power(np.arange(N), a))
    fhistory = np.zeros((N - 1, d), dtype=type(y0[0]))
    y = np.zeros((N, d), dtype=type(y0[0]))
    y[0] = y0;
    for n in range(0, N - 1):
        tn = tspan[n]
        yn = y[n]
        fhistory[n] = f(yn, tn)
        y[n+1] = y0 + np.dot(w[0:n+1], fhistory[n::-1])
    return y
