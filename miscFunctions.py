#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
#from scipy import integrate

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

def logNormal(mu, sig, r=None):
    """
    logNormal: (float, float, array*) -> (array, array)
    Parameters:
        mu: median radius (this is exp(mu) at https://en.wikipedia.org/wiki/Log-normal_distribution)
        sig: regular (not geometric) sigma
        r: optional array of radii at which to return dX/dr 
    Returns tupple with two arrays (dX/dr, r) 
    """
    if r is None:
        Nr = 1e4 # number of radii
        Nfct = 5 # r spans this many geometric std factors above and below mu
        bot = np.log10(mu) - Nfct*sig/np.log(10)
        top = np.log10(mu) + Nfct*sig/np.log(10)
        r = np.logspace(bot,top,Nr)
    nrmFct = 1/(sig*np.sqrt(2*np.pi))
    dxdr = nrmFct*(r**-1)*np.exp(-((np.log(r)-np.log(mu))**2)/(2*sig**2))
    return dxdr,r

def effRadius(r, dvdlnr):
    vol = np.trapz(dvdlnr/r,r)
    area = np.trapz(dvdlnr/r**2,r)
    return vol/area
        
        
        
        
        