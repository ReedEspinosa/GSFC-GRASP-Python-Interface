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