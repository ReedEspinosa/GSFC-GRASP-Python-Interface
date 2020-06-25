#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
try:
    import PyMieScatt as ps
    PyMieLoaded = True
except ImportError:
    PyMieLoaded = False
try:
    import matplotlib.pyplot as plt
    pltLoad = True
except ImportError:
    pltLoad = False
#from scipy import integrate

def checkDiscover(): # right now this just checks for a remote connection...
    return "SSH_CONNECTION" in os.environ

def norm2absExtProf(normalizedProfile, heights, AOD):
    """
    normalizedProfile -> 1xN array like; normalized profile
    heights -> 1xN array like;; altitude bins, must be sorted (both descending and ascending are fine)
    AOD -> scalar; AOD value for the mode
    """
    from scipy.integrate import simps
    C = np.abs(simps(normalizedProfile, heights))
    return AOD*normalizedProfile/C
    
    

def matplotlibX11():
    if checkDiscover():
        import matplotlib
        matplotlib.use('TkAgg')

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
        Nr = int(1e4) # number of radii
        Nfct = 5 # r spans this many geometric std factors above and below mu
        bot = np.log10(mu) - Nfct*sig/np.log(10)
        top = np.log10(mu) + Nfct*sig/np.log(10)
        r = np.logspace(bot,top,Nr)
    nrmFct = 1/(sig*np.sqrt(2*np.pi))
    dxdr = nrmFct*(r**-1)*np.exp(-((np.log(r)-np.log(mu))**2)/(2*sig**2))
    return dxdr,r # random note: rEff = mu*np.exp(-sig**2/2)

def effRadius(r, dvdlnr):
    vol = np.trapz(dvdlnr/r,r)
    area = np.trapz(dvdlnr/r**2,r)
    return vol/area
        
def phaseMat(r, dvdlnr, n, k, wav=0.550):
    """
    # https://pymiescatt.readthedocs.io
    #SR = |S1|^2   # we lose information here so there is no way to recover S33 or S34...
    #SL = |S2|^2   # there is a function that finds these last two elements but only for monodisperse PSD
    #SU = 0.5(SR+SL)
    #S11 = 0.5 (|S2|^2 + |S1|^2)
    #S12 = 0.5 (|S2|^2 - |S1|^2)  [S12/S11=1 -> unpolarized light is scattered into 100% polarized light oriented perpendicular to scattering plane]
    phaseMat: (array, array, float, float, float*) -> (array, array, array)
    Parameters:
        r: array of radii in μm
        dvdlnr: array of PSD values at r in dv/dlnr
        n: real refractive index
        k: imaginary refracrtive index
        wav: wavelength in μm
    Returns tupple with three arrays: (scattering_angle, normalized_P11, -P12/P11) 
    """
    assert PyMieLoaded, "Import errors occured when loading the PyMieScatt module" 
    m = complex(n, k)   
    dp = r*2
    ndp = dvdlnr/(r**3) # this should be r^4 but we seem to match with r^3... we _think_ PyMieScatt really wants dn/dr*r, contrary to docs
    theta,sl,sr,su = ps.SF_SD(m, wav, dp, ndp, angularResolution=1)
    S11=0.5*(sl+sr)
    S12=-0.5*(sl-sr) # minus makes positive S12 polarized in scattering plane
    p11 = 2*S11/np.trapz(S11*np.sin(theta), theta)
    return theta*180/np.pi, p11, -S12/S11
        
def gridPlot(xlabel, ylabel, values):
    assert pltLoad, 'Matplotlib could not be loaded!'
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(15,6), frameon=False)
    ax.imshow(np.sqrt(values), 'seismic', vmin=0, vmax=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xlabel)))
    ax.set_yticks(np.arange(len(ylabel)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xlabel)
    ax.set_yticklabels(ylabel)
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(ylabel)):
        for j in range(len(xlabel)):
            valStr = '%3.1f' % values[i, j]
            clr = 'w' if np.abs(values[i, j]-1)>0.5 else 'k'
    #        clr = np.min([np.abs(harvest[i, j]-1)**3, 1])*np.ones(3)
            ax.text(j, i, valStr,
                           ha="center", va="center", color=clr, fontsize=9)
    fig.tight_layout()   
        
#TODO: should make PSD class with r,dNdr and type (dndr,dvdr,etc.)      
