#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from simulateRetrieval import simulation
from glob import glob

waveInd = 2
waveInd2 = 4
fnPtrnList = []
fnPtrnList.append('Basic01_modis_HuamboVegetation_tFct1.000_n0_nAng*.pkl')
fnPtrnList.append('MultiPix10_modis_HuamboVegetation_tFct1.000_n0_nAng*.pkl')
# fnPtrnList.append('MultiPix20_modis_HuamboVegetation_tFct1.000_n0_nAng*.pkl')
colorList = [[0.7,0.7,0.95], [0.9,0.0,0.0]]
inDirPath = '/Users/wrespino/synced/Working/TASNPP_simulation00/'

biasCorrect = 0

xlabel = 'Simulated Truth'
MS = 1
FS = 10
LW121 = 1
fig, ax = plt.subplots(1,3, figsize=(9.4,3.3))

for i,(fnPtrn,clrNow) in enumerate(zip(fnPtrnList,colorList)):
    saveFN = 'MERGED_'+fnPtrn.replace('*','ALL')
    files = glob(os.path.join(inDirPath, fnPtrn))
    if len(files)>0:
        simBase = simulation()
        simBase.rsltFwd = np.empty(0, dtype=dict)
        simBase.rsltBck = np.empty(0, dtype=dict)
        print('Building %s - Nfiles=%d' % (saveFN, len(files)))
        for file in files: # loop over all available nAng            
            simA = simulation(picklePath=file)
            NrsltBck = len(simA.rsltBck)
            print('%s - %d' % (file, NrsltBck))
            if NrsltBck==32:
                Nrepeats = 1 if NrsltBck==len(simA.rsltFwd) else NrsltBck
                for _ in range(Nrepeats): simBase.rsltFwd = np.r_[simBase.rsltFwd, simA.rsltFwd]
                simBase.rsltBck = np.r_[simBase.rsltBck, simA.rsltBck]
            else:
                print('   Length not 32... Skipping this entry.')
        simBase.saveSim(os.path.join(inDirPath,saveFN))
        print('Saving to %s - %d' % (saveFN, len(simBase.rsltBck)))
    else:
        print('No files found')
    print('--')    

    simOutputPath = os.path.join(inDirPath, saveFN)
    sim = simulation(picklePath=simOutputPath)
    pprint(sim.analyzeSim(waveInd)[0])

    shft = ''.join([' ' for _ in range((i+1)*20)])
    
    # AOD
    true = np.asarray([rf['aod'][waveInd] for rf in sim.rsltFwd])
    rtrv = np.asarray([rf['aod'][waveInd] for rf in sim.rsltBck])
    if i==0:
        minAOD = np.min(true)*0.95
        maxAOD = np.max(true)*1.05
        ax[0].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
        ax[0].set_title('AOD')
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel('Retrieved')
        ax[0].set_xlim(minAOD,maxAOD)
        ax[0].set_ylim(minAOD,maxAOD)
    ax[0].plot(true, rtrv, '.', markersize=MS, color=clrNow)
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    if i==len(colorList)-1:
        frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
        tHnd = ax[0].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(115, -147), va='top', xycoords='axes fraction',
                    textcoords='offset points', color='k', fontsize=FS)
    else:
        frmt = shft+'(%5.3f)\n'+shft+'(%5.3f)\n'+shft+'(%5.3f)'
    textstr = frmt % (Rcoef, RMSE, bias)
    tHnd = ax[0].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                        textcoords='offset points', color=clrNow, fontsize=FS)
    if i==len(colorList)-2: tHnd.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='None'))
        
    # ANGSTROM
    aod2 = np.asarray([rf['aod'][waveInd2] for rf in sim.rsltFwd])
    logLamdRatio = np.log(sim.rsltFwd[0]['lambda'][waveInd]/sim.rsltFwd[0]['lambda'][waveInd2])
    true = -np.log(true/aod2)/logLamdRatio
    aod2 = np.asarray([rf['aod'][waveInd2] for rf in sim.rsltBck])
    rtrv = -np.log(rtrv/aod2)/logLamdRatio 
    rtrv = rtrv - 0.6*(2.2-rtrv)*(i==1)*biasCorrect-0.1
    if i==0:
        minAOD = np.min(true)*0.95
        minAOD = 1.6
        maxAOD = np.max(true)*1.05
        maxAOD = 2.41
        ax[1].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
        ax[1].set_title('Angstrom Exponent')
        ax[1].set_xlabel(xlabel)
        ax[1].set_xlim(minAOD,maxAOD)
        ax[1].set_ylim(minAOD,maxAOD)
        ax[1].set_xticks(np.arange(minAOD, maxAOD, 0.2))
        ax[1].set_yticks(np.arange(minAOD, maxAOD, 0.2))
    ax[1].plot(true, rtrv, '.', markersize=MS, color=clrNow)
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    textstr = frmt % (Rcoef, RMSE, bias)
    tHnd = ax[1].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                        textcoords='offset points', color=clrNow, fontsize=FS)
    if i==len(colorList)-2: tHnd.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='None'))
    
    # SSA
    true = np.asarray([rf['ssa'][waveInd] for rf in sim.rsltFwd])
    rtrv = np.asarray([rf['ssa'][waveInd] for rf in sim.rsltBck]) 
    if biasCorrect>0 and i==1:
        rtrv[rtrv<0.925] = rtrv[rtrv<0.925] - 0.022
        indMid = np.logical_and(rtrv>=0.925, rtrv<0.940)
        rtrv[indMid] = rtrv[indMid] - 0.022*(0.940-rtrv[indMid])/0.015
    if i==0:
        minAOD = np.min(true)*0.95
        minAOD = 0.735
        maxAOD = 1
        ax[2].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
        ax[2].set_title('SSA')
        ax[2].set_xlabel(xlabel)
        ax[2].set_xlim(minAOD,maxAOD)
        ax[2].set_ylim(minAOD,maxAOD)
        ax[2].set_xticks(np.arange(0.75, 1.01, 0.05))
        ax[2].set_yticks(np.arange(0.75, 1.01, 0.05))
    ax[2].plot(true, rtrv, '.', markersize=MS, color=clrNow)
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    textstr = frmt % (Rcoef, RMSE, bias)
    tHnd = ax[2].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                        textcoords='offset points', color=clrNow, fontsize=FS)
    if i==len(colorList)-2: tHnd.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='None'))
    
plt.tight_layout()
plt.ion()
plt.show()




