#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:40:41 2019

@author: wrespino
"""

import numpy as np
import os
import sys
sys.path.append(os.path.join(".."))
import runGRASP as rg
import simulateRetrieval as rs

fwdModelYAMLpath = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_5lambda_CASE-6a-onlyMARINE_V0.yml'
bckYAMLpath = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_5lambda_Template.yml'
savePath = '/Users/wrespino/Desktop/testCase_PolMISR_6aMARINE.pkl'
Nsims = 20
maxCPU = 2

# DUMMY MEASUREMENTS (determined by architecture, should ultimatly move to seperate scripts)
#  For more than one measurement type or viewing geometry pass msTyp, nbvm, thtv, phi and msrments as vectors: \n\
#  len(msrments)=len(thtv)=len(phi)=sum(nbvm); len(msTyp)=len(nbvm) \n\
#  msrments=[meas[msTyp[0],thtv[0],phi[0]], meas[msTyp[0],thtv[1],phi[1]],...,meas[msTyp[0],thtv[nbvm[0]],phi[nbvm[0]]],meas[msTyp[1],thtv[nbvm[0]+1],phi[nbvm[0]+1]],...]'
measNm = ['I', 'Q', 'U'] # should match GRASP output (e.g. fit_I -> 'I')
msTyp = [41, 42, 43]
nbvm = [9, 9, 9]
sza = 30
thtv = np.tile([70.5, 60.0, 45.6, 26.1, 0, 26.1, 45.6, 60.0, 70.5], len(nbvm))
phi = np.tile([0, 0, 0, 0, 0, 180, 180, 180, 180], len(nbvm))
meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])] 
nowPix = rg.pixel(730123.0, 1, 1, 0, 0, 0, 100)
nowPix.addMeas(0.36, msTyp, nbvm, sza, thtv, phi, meas)
nowPix.addMeas(0.55, msTyp, nbvm, sza, thtv, phi, meas)
nowPix.addMeas(0.87, msTyp, nbvm, sza, thtv, phi, meas)
nowPix.addMeas(1.23, msTyp, nbvm, sza, thtv, phi, meas)
nowPix.addMeas(1.65, msTyp, nbvm, sza, thtv, phi, meas)
nbvm = np.tile(nbvm, [len(nowPix.measVals),1]) # this assumes nbvm(wvlth,msType) is same at all lambda
measNm = np.tile(measNm, [len(nowPix.measVals),1]) # this assumes nbvm(wvlth,msType) is same at all lambda

def addError(meas, measNm):
    if measNm=='I':
        return meas*(1+np.random.normal()*0.03)
    elif measNm=='Q' or measNm=='U':
        return meas*(1+np.random.normal()*0.005)
    else:
        assert False, 'Unknown measurement string, can not add error!'

simA = rs.simulation(nowPix, addError, measNm) # defines new instance for this architecture
simA.runSim(fwdModelYAMLpath, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath) # runs the simulation for given set of conditions 
simA.analyzeSim()
# NOTE: this last line ultimatly could be loop over all canonical cases OR various numbers of simultations to test convergance
