#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:11:26 2019

@author: wrespino
"""

import numpy as np
import matplotlib.pyplot as plt
from simulateRetrieval import simulation
#Working/SIM13_lidarTest/SIM42_lidar05+polar07_case-variable_sza15_phi0_tFct0.20_V2.pkl
#simRsltFile = '/Users/wrespino/Synced/Working/SIM8_pre613Seminar/SIM_lidar05+img02_case-variable_sza30_phi0_tFct0.35_V2.pkl'
simRsltFile = '/Users/wrespino/Synced/Working/SIM13_lidarTest/SIM43_lidar05_case-case06fmonomode_sza30_phi0_tFct1.00_V2.pkl'
#simRsltFile = '/Users/wrespino/Synced/Working/SIM13_lidarTest/SIM25_lidar05+polar07_case-case06amonomode_sza30_phi0_tFct1.00_V2.pkl'
lInd = 0
modeInd = 0

βfun = lambda i,l,d: d['aodMode'][i,l]*d['βext'][i,:]/np.mean(d['βext'][i,:])

simA = simulation(picklePath=simRsltFile)
alphVal = 1/np.sqrt(len(simA.rsltBck))
color1 = np.r_[1, 0, 0]
color2 = np.r_[0, 0, 1]
profExtNm = 'βext'
hsrl = 'fit_VExt' in simA.rsltFwd
figA, axA = plt.subplots(1,2+hsrl,figsize=(9+3*hsrl,6))
measTypes = ['VExt', 'VBS'] if hsrl else ['LS']
for rb in simA.rsltBck:
#    [np.sum(rb[profExtNm][i,:]*rb['range'][i,:]) for i in range(rb['range'].shape[0])]
#    botPrf = 0 if np.sum(rb[profExtNm][0,:]*rb['range'][0,:])>np.sum(rb[profExtNm][1,:]*rb['range'][1,:]) else 1
#    axA[0].plot(βfun(botPrf%2,lInd,rb), rb['range'][botPrf%2,:]/1e3, color=color1, alpha=alphVal)
#    axA[0].plot(βfun((botPrf+1)%2,lInd,rb), rb['range'][(botPrf+1)%2,:]/1e3, color=color2, alpha=alphVal)
    axA[0].plot(βfun(0,lInd,rb), rb['range'][0,:]/1e3, color=color1, alpha=alphVal)
    axA[0].plot(βfun(1,lInd,rb), rb['range'][1,:]/1e3, color=color2, alpha=alphVal)
    for i,mt in enumerate(measTypes):
        axA[i+1].plot(1e6*rb['meas_'+mt][:,lInd], rb['RangeLidar'][:,lInd]/1e3, color=color1, alpha=alphVal)
        axA[i+1].plot(1e6*rb['fit_'+mt][:,lInd], rb['RangeLidar'][:,lInd]/1e3, color=color2, alpha=alphVal)
md1Hnd = axA[0].plot(βfun(0,lInd,simA.rsltFwd), simA.rsltFwd['range'][0,:]/1e3, 'o', color=color1/2)
md2Hnd = axA[0].plot(βfun(1,lInd,simA.rsltFwd), simA.rsltFwd['range'][1,:]/1e3, 'o', color=color2/2)
for i,mt in enumerate(measTypes):
    axA[i+1].plot(1e6*simA.rsltFwd['fit_'+mt][:,lInd], simA.rsltFwd['RangeLidar'][:,lInd]/1e3, 'ko')
    axA[i+1].legend(['Measured', 'Retrieved'])
    axA[i+1].set_xlim([0,2*1e6*simA.rsltFwd['fit_'+mt][:,lInd].max()])
axA[0].legend(md1Hnd+md2Hnd,['Mode 1', 'Mode 2'])
axA[0].set_ylabel('Altitude (km)')
axA[0].set_xlabel('Extinction (normalized)')
#axA[0].set_xlim([0,1])
if hsrl:
    axA[1].set_xlabel('Extinction ($Mm^{-1}$)')
    axA[2].set_xlabel('Backscatter ($Mm^{-1}Sr^{-1}$)')
else:
    axA[1].set_xlabel('Attenuated Backscatter ($Mm^{-1}Sr^{-1}$)')

#plt.figure()
#plt.plot(simA.rsltFwd['fit_VExt'][:,lInd]/simA.rsltFwd['fit_VBS'][:,lInd], simA.rsltFwd['RangeLidar'][:,lInd]/1e3, 'k')



