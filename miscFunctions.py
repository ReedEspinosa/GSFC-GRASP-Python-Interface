#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def angstrmIntrp(lmbdIn, tau, lmbdTrgt):
    tau = tau[lmbdIn.argsort()]
    lmbd = lmbdIn[lmbdIn.argsort()]
#    assert (lmbdTrgt >= lmbd.min() and lmbdTrgt <= lmbd.max()), "This function will not extroplate, lmbdTrgt must fall between two values."
    if lmbdTrgt < lmbd.min() or lmbdTrgt > lmbd.max():
        return np.nan
    if lmbdTrgt==lmbd[0]:
        return tau[0]
    if lmbdTrgt==lmbd[-1]:
        return tau[-1]
    frstInd = np.nonzero((lmbd - lmbdTrgt) < 0)[0][-1]
    alpha = angstrm(lmbd[frstInd:frstInd+2], tau[frstInd:frstInd+2])
    return tau[frstInd]*(lmbd[frstInd]/lmbdTrgt)**alpha
    
def angstrm(lmbd, tau):
    assert (lmbd.shape[0]==2 and tau.shape[0]==2), "Exactly two values must be provided!"
    return -np.log(tau[0]/tau[1])/np.log(lmbd[0]/lmbd[1])
    
def simpsonsRule(f,a,b,N=50):
    """
    simpsonsRule: (func, array, int, int) -> float
    Parameters:
		f: function that returns the evaluated equation at point x.
        a, b: integers representing lower and upper bounds of integral.
		N: integers number of segments being used to approximate the integral (same n as http://en.wikipedia.org/wiki/Simpson%27s_rule)
	Returns float equal to the approximate integral of f(x) from bnds[0] to bnds[1] using Simpson's rule.
    """
    assert np.mod(N,2)==0, 'n must be even!'
    dx = (b-a)/N
    x = np.linspace(a,b,N+1)
    y = f(x)
    S = dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])
    return S
    