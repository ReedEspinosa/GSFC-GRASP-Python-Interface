#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:11:26 2019

@author: wrespino
"""

import numpy as np
import matplotlib.pyplot as plt
from simulateRetrieval import simulation

simRsltFile = '/Users/wrespino/Synced/Working/SIM8/SIM_lidar05+img02_case-variable_sza0_phi0_tFct0.35_V2.pkl'
lInd = 1
modeInd = 0

simA = simulation(picklePath=simRsltFile)
alphVal = 1/np.sqrt(len(simA.rsltBck))
color1 = np.r_[1, 0, 0]
color2 = np.r_[0, 0, 1]
profExtNm = 'βext'
hsrl = 'fit_VExt' in simA.rsltFwd
figA, axA = plt.subplots(1,2+hsrl,figsize=(9+3*hsrl,6))
measTypes = ['VExt', 'VBS'] if hsrl else ['LS']
for rb in simA.rsltBck:
    axA[0].plot(rb[profExtNm][0,:]/np.mean(rb[profExtNm][0,:]), rb['range'][0,:]/1e3, color=color1, alpha=alphVal)
    axA[0].plot(rb[profExtNm][1,:]/np.mean(rb[profExtNm][1,:]), rb['range'][1,:]/1e3, color=color2, alpha=alphVal)
    for i,mt in enumerate(measTypes):
        axA[i+1].plot(1e6*rb['meas_'+mt][:,lInd], rb['RangeLidar'][:,lInd]/1e3, color=color1, alpha=alphVal)
        axA[i+1].plot(1e6*rb['fit_'+mt][:,lInd], rb['RangeLidar'][:,lInd]/1e3, color=color2, alpha=alphVal)
md1Hnd = axA[0].plot(simA.rsltFwd[profExtNm][0,:]/np.mean(simA.rsltFwd[profExtNm][0,:]), simA.rsltFwd['range'][0,:]/1e3, color=color1/2)
md2Hnd = axA[0].plot(simA.rsltFwd[profExtNm][1,:]/np.mean(simA.rsltFwd[profExtNm][1,:]), simA.rsltFwd['range'][1,:]/1e3, color=color2/2)
for i,mt in enumerate(measTypes):
    axA[i+1].plot(1e6*simA.rsltFwd['fit_'+mt][:,lInd], simA.rsltFwd['RangeLidar'][:,lInd]/1e3, 'k')
    axA[i+1].legend(['Measured', 'Retrieved'])
    axA[i+1].set_xlim([0,2*1e6*simA.rsltFwd['fit_'+mt][:,lInd].max()])
axA[0].legend(md1Hnd+md2Hnd,['Mode 1', 'Mode 2'])
axA[0].set_ylabel('Altitude (km)')
axA[0].set_xlabel('Extinction (normalized)')
axA[0].set_xlim([0,5])
if hsrl:
    axA[1].set_xlabel('Extinction ($Mm^{-1}$)')
    axA[2].set_xlabel('Backscatter ($Mm^{-1}Sr^{-1}$)')
else:
    axA[1].set_xlabel('Attenuated Backscatter ($Mm^{-1}Sr^{-1}$)')