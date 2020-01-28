#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import datetime as dt
sys.path.append(os.path.join(".."))
import runGRASP as rg
import simulateRetrieval as rs
import functools

# MacBook Air
#fwdModelYAMLpath = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_5lambda_CASE-6a-onlyMARINE_V0.yml'
#bckYAMLpath = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_5lambda_Template.yml'
#savePath = '/Users/wrespino/Desktop/testCase_PolMISR_6aMARINE.pkl'
#dirGRASP = None
#krnlPath = None
#Nsims = 4
#maxCPU = 2

# DISCOVER
basePath = os.environ['NOBACKUP']
dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
fwdModelYAMLpath = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_5lambda_CASE-6a-onlyMARINE_V0.yml')
bckYAMLpath = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_5lambda_Template.yml')
savePath = os.path.join(basePath, 'synced/Working/testDISCOVER_PolMISR_6aMARINE_Sep7_V1.pkl')
Nsims = 84
maxCPU = 28

def addError(measNm, l, rsltFwd, edgInd):
    assert measNm==41, 'Something went wrong with functools!' #HACK
    # if the following check ever becomes a problem see commit #6793cf7
    assert (np.diff(edgInd)[0]==np.diff(edgInd)).all(), 'Current error models assume that each measurement type has the same number of measurements at each wavelength!'
    relErr = 0.03
    relDoLPErr = 0.005
    trueSimI = rsltFwd['fit_I'][:,l]
    trueSimQ = rsltFwd['fit_Q'][:,l]
    trueSimU = rsltFwd['fit_U'][:,l]
    noiseVctI = np.random.lognormal(sigma=np.log(1+relErr), size=len(trueSimI))
    fwdSimI = trueSimI*noiseVctI
    fwdSimQ = trueSimQ*noiseVctI # we scale Q and U too to keep q, u and DoLP inline with truth
    fwdSimU = trueSimU*noiseVctI
    dpRnd = np.random.normal(size=len(trueSimI))*relDoLPErr
    dPol = dpRnd*trueSimI*np.sqrt((trueSimQ**2+trueSimU**2)/(trueSimQ**4+trueSimU**4)) # true is fine b/c noiceVctI factors cancel themselves out
    fwdSimQ = fwdSimQ*(1+dPol)
    fwdSimU = fwdSimU*(1+dPol) # Q and U errors are 100% correlated here
    fwdSim = np.r_[fwdSimI, fwdSimQ, fwdSimU] # safe because of ascending order check in simulateRetrieval.py
    return fwdSim

# DUMMY MEASUREMENTS (determined by architecture, should ultimatly move to seperate scripts)
#  For more than one measurement type or viewing geometry pass msTyp, nbvm, thtv, phi and msrments as vectors: \n\
#  len(msrments)=len(thtv)=len(phi)=sum(nbvm); len(msTyp)=len(nbvm) \n\
#  msrments=[meas[msTyp[0],thtv[0],phi[0]], meas[msTyp[0],thtv[1],phi[1]],...,meas[msTyp[0],thtv[nbvm[0]],phi[nbvm[0]]],meas[msTyp[1],thtv[nbvm[0]+1],phi[nbvm[0]+1]],...]'
msTyp = [41, 42, 43] # must be in ascending order
nbvm = 9*np.ones(len(msTyp), np.int)
sza = 30
relPhi = 0
thtv = np.tile([-70.5, -60.0, -45.6, -26.1, 0, 26.1, 45.6, 60.0, 70.5], len(msTyp))
wvls = [0.36, 0.55, 0.87, 1.23, 1.65] # Nλ=5
meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])] 
phi = np.repeat(relPhi, len(thtv)) # currently we assume all observations fall within a plane
nowPix = rg.pixel(dt.datetime.now(), 1, 1, 0, 0, 0, 100)
for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
    nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas, functools.partial(addError, 41)) # this must link to an error model in addError() above 

# RUN SIMULATION
simA = rs.simulation(nowPix) # defines new instance for this architecture
# runs the simulation for given set of conditions, releaseYAML=True -> index of wavelength involved YAML fields MUST cover every wavelength BUT bckYAML Nλ does not have to match fwd calulcation
simA.runSim(fwdModelYAMLpath, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True) 
rmsErr, meanBias = simA.analyzeSim()
print(rmsErr)
print(meanBias)
