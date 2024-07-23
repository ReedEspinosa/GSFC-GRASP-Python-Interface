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
# from scipy import integrate
from random import random
from math import log

def loguniform(lo,hi):
    '''
    Definition to create the random number that is uniformly spaced between 2 values
    '''
    if isinstance(lo,(list,np.ndarray)):
        if len(lo) > 1:
            logRnd =[]
            logRnd_ = lo[0] ** ((((log(hi[0]) / log(lo[0])) - 1) * random()) + 1) # Spectrally flat #HACK, it can be changed in the future
            for i in range(0,len(lo)):
                logRnd.append(logRnd_)
            
            return logRnd
        else:
            return lo ** ((((log(hi) / log(lo)) - 1) * random()) + 1)
    else:
        return lo ** ((((log(hi) / log(lo)) - 1) * random()) + 1)

def slope4RRI(RRI, wavelengths, slope=0.01):
    '''
    Add a slope to the RRI
    '''
    return RRI - slope*RRI*wavelengths

def loguniform(lo,hi):
    '''
    Definition to create the random number that is uniformly spaced between 2 values
    '''
    if isinstance(lo,(list,np.ndarray)):
        if len(lo) > 1:
            logRnd =[]
            logRnd_ = lo[0] ** ((((log(hi[0]) / log(lo[0])) - 1) * random()) + 1) # Spectrally flat #HACK, it can be changed in the future
            for i in range(0,len(lo)):
                logRnd.append(logRnd_)
            
            return logRnd
        else:
            return lo ** ((((log(hi) / log(lo)) - 1) * random()) + 1)
    else:
        return lo ** ((((log(hi) / log(lo)) - 1) * random()) + 1)

def slope4RRI(RRI, wavelengths, slope=0.01):
    '''
    Add a slope to the RRI
    '''
    return RRI - slope*RRI*wavelengths

def checkDiscover(): # right now this just checks for a remote connection...
    return "SSH_CONNECTION" in os.environ


def seaLevelROD(λtarget):
    """ Returns the ROD at λtarget wavelength at sea level """
    λtarget = np.asarray(λtarget)
    λ =   np.r_[0.3600, 0.3800, 0.4100, 0.5500, 0.6700, 0.8700, 1.5500, 1.6500]
    rod = np.r_[0.5612, 0.4474, 0.3259, 0.0973, 0.0436, 0.0152, 0.0015, 0.0012]
    assert λ.min()<=λtarget.min() and λ.max()>=λtarget.max(), 'λtarget falls outside the range of pre-programed values!'
    return np.interp(λtarget, λ, rod**-0.25)**-4


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
    if lmbdTrgt <= lmbd.min(): # calculate α from lowest two λ
        frstInd = 0
    elif lmbdTrgt >= lmbd.max(): # calculate α from highest two λ
        frstInd = len(lmbd)-2
    else: # calculate α from adjacent two (above and below) λ
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
    dxdr = nrmFct*(1/r)*np.exp(-((np.log(r)-np.log(mu))**2)/(2*sig**2))
    return dxdr,r # random note: rEff = mu*np.exp(-sig**2/2)


def calculatePM(rsltList, upperSize=2.5, rho_mass=1, alt=2):
    Nmodes = rsltList[0]['aodMode'].shape[0]
    Nrslts = len(rsltList)
    grndConcFrac = np.empty((Nrslts, Nmodes))
    totVolConc = np.empty((Nrslts, Nmodes))
    for mode in range(Nmodes):
         grndConcFrac[:,mode] = integrateProfile(rsltList, mode, lowBnd=alt-0.5, upBnd=alt+0.5)
         totVolConc[:,mode] = integratePSD(rsltList, upBnd=upperSize, sizeMode=mode)
    particulateVol = np.sum(grndConcFrac*totVolConc, axis=1)
    return rho_mass*1e6*particulateVol # return in μg/m^3; rho_mass should have units of g/cm^3 (e.g. rho_mass(H2O)=1)


def integrateProfile(rsltList, mode, lowBnd=1.5, upBnd=2.5):
    # Single mode retrievals may break this function (probably could be fixed with some atleast_Nd's below)
    if 'heightStd' in rsltList[0]: from scipy.stats import norm as spyNorm
    output = np.empty(len(rsltList))
    for i,rs in enumerate(rsltList):
        if 'height' in rs.keys() and 'heightStd' in rs.keys():
            hgt = rs['height'][mode]
            std = rs['heightStd'][mode]
            output[i] = spyNorm.cdf(upBnd,scale=std,loc=hgt) - spyNorm.cdf(lowBnd,scale=std,loc=hgt)
        elif 'βext' in rs.keys():
            if rs['range'].shape[0]<=mode: # TODO: This is ugly and could lead to unexpected behavior but Nmodes varies in some canonical cases; needs better handling though
                output[i] = 0
            else:
                rng = rs['range'][mode,::-1]
                ext = rs['βext'][mode,::-1]
                assert np.all(rng[:-1] <= rng[1:]), 'Extinction profile and range arrays should be increasing from the ground (range=0) up.'
                if min(rng)<lowBnd or max(rng)>upBnd: # We only integrate within existing vertical profile; i.e., βext=0 for z<[z_grasp] and z>[z_grasp]
                    if max(rng) <= upBnd: # only trim low end of radii
                        rngNew = np.r_[lowBnd, rng[rng>lowBnd]]
                    elif min(rng) >= lowBnd: # only trim high end of radii
                        rngNew = np.r_[rng[r<upBnd], upBnd]   
                    else: # trim low and high ends of radii            
                        midIndKeep = np.logical_and(rng>lowBnd, rng<upBnd)
                        rngNew = np.r_[lowBnd, rng[midIndKeep], upBnd]
                indKeep = np.logical_and(rng>lowBnd, rng<upBnd)
                ext = np.interp(rngNew, rng, ext, left=0, right=0)
                λ550Ind = np.argmin(np.abs(rs['lambda']-0.55)) # in runGRASP we scale βext to 1/Mm at λ=550nm (or next closest λ)
                output[i] = np.trapz(ext, rngNew)/1e6/rs['aodMode'][mode, λ550Ind] # units of βext are 1/Mm; we convert to 1/m
        elif 'height' in rs.keys():
            assert False, 'This is likely exp profile which integrateProfile() does not currently support.'
        else:
            assert False, 'Could not determine profile type present in this rsltList.'
    return output

def integratePSD(rsltList, moment='vol', lowBnd=0, upBnd=np.inf, sizeMode=None, verbose=False):
    import scipy.stats
    # volume returned in μm3/μm2; conveniently, this is also g/m3 for ρ_mass_H20
    if moment.lower()=='reff': 
        vol = integratePSD(rsltList, 'vol', lowBnd, upBnd) # we actually want int(area*r*dr)=3*V for area weighted radius but...
        area = integratePSD(rsltList, 'area_pure', lowBnd, upBnd) # pure argument here skips the 3x in area calculation so the two cancel in ratio
        return vol/area
    output = np.empty(len(rsltList))
    for i,rs in enumerate(rsltList):
        if sizeMode is None: # we want to sum over them all
            if not np.all(rs['r'][0]==rs['r']):
                if verbose: print('Warning assumption that all modes specfified over the same radii was violated! Returning list of np.nan...')
                return np.full(len(rsltList), np.nan)
            r = rs['r'][0]
            dVdlnr = rs['dVdlnr'].sum(axis=0)
        elif rs['r'].shape[0]>sizeMode: # we take dVdlnr from just one of the modes
            r = rs['r'][sizeMode]
            dVdlnr = rs['dVdlnr'][sizeMode]
        else: # TODO: This is ugly and could lead to unexpected behavior but Nmodes varies in some canonical cases; needs better handling though
            r = rs['r'][0]
            dVdlnr = np.zeros(len(r))
        if min(r)<lowBnd or max(r)>upBnd: # We only integrate within existing PSD; i.e., PSD=0 for r<r_grasp and r>r_grasp
            if max(r) <= upBnd: # only trim low end of radii
                rNew = np.r_[lowBnd, r[r>lowBnd]]
            elif min(r) >= lowBnd: # only trim high end of radii
                rNew = np.r_[r[r<upBnd], upBnd]            
            else: # trim low and high ends of radii            
                midIndKeep = np.logical_and(r>lowBnd, r<upBnd)
                rNew = np.r_[lowBnd, r[midIndKeep], upBnd]
            dVdlnr = np.interp(rNew, r, dVdlnr)
            r = rNew
        if moment.lower()=='vol':
            output[i] = np.trapz(dVdlnr/r,r)
        elif moment.lower()=='area' or moment.lower()=='area_pure':
            area = np.trapz(dVdlnr/r**2,r)
            output[i] = area if 'pure' in moment else 3*area
        elif moment.lower()=='num':
            output[i] = 3/4/np.pi*np.trapz(dVdlnr/r**4,r)
        else:
            assert False, "%s moment not recognized. Options are 'vol', 'area', 'num', or 'reff'." % moment
    return output

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

# TODO: should make PSD class with r,dNdr and type (dndr,dvdr,etc.)
