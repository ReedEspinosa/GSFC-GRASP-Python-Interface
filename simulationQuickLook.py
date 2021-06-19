#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from simulateRetrieval import simulation
from glob import glob

waveInd = 3
waveInd2 = 5
fnPtrnList = []
fnPtrn = 'ss450-g5nr.leV120.GRASP.example.polarimeter07.20060802_*z.pkl'
inDirPath = '/Users/wrespino/Synced/Working/OSSE_Test_Run'
surf2plot = 'both' # land, ocean or both
aodMin = 0.0

xlabel = 'Simulated Truth'
MS = 1
FS = 10
LW121 = 1
pointAlpha = 0.15
clrText = [0.5,0,0.0]
fig, ax = plt.subplots(2,3, figsize=(10.4,6.3))


saveFN = 'MERGED_'+fnPtrn.replace('*','ALL')
savePATH = os.path.join(inDirPath,saveFN)
if os.path.exists(savePATH):
    simBase = simulation(picklePath=savePATH)
    print('Loading from %s - %d' % (saveFN, len(simBase.rsltBck)))
else:
    files = glob(os.path.join(inDirPath, fnPtrn))
    assert len(files)>0, 'No files found!'
    simBase = simulation()
    simBase.rsltFwd = np.empty(0, dtype=dict)
    simBase.rsltBck = np.empty(0, dtype=dict)
    print('Building %s - Nfiles=%d' % (saveFN, len(files)))
    for file in files: # loop over all available nAng
        simA = simulation(picklePath=file)
        NrsltBck = len(simA.rsltBck)
        print('%s - %d' % (file, NrsltBck))
        Nrepeats = 1 if NrsltBck==len(simA.rsltFwd) else NrsltBck
        for _ in range(Nrepeats): simBase.rsltFwd = np.r_[simBase.rsltFwd, simA.rsltFwd]
        simBase.rsltBck = np.r_[simBase.rsltBck, simA.rsltBck]
    simBase.saveSim(savePATH)
    print('Saving to %s - %d' % (saveFN, len(simBase.rsltBck)))
print('--')

# print general stats to console
print('Showing results for %5.3f μm' % simBase.rsltFwd[0]['lambda'][waveInd])
pprint(simBase.analyzeSim(waveInd)[0])
lp = np.array([rf['land_prct'] for rf in simBase.rsltFwd])
keepInd = lp>99 if surf2plot=='land' else lp<1 if surf2plot=='ocean' else lp>-1
keepInd = np.logical_and(keepInd, [rf['aod'][waveInd]>aodMin for rf in simBase.rsltFwd])

# variable to color point by in all subplots
clrVar = np.array([rb['costVal'] for rb in simBase.rsltBck])

# AOD
true = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
ax[0,0].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[0,0].set_title('AOD')
ax[0,0].set_xlabel(xlabel)
ax[0,0].set_ylabel('Retrieved')
ax[0,0].set_xlim(minAOD,maxAOD)
ax[0,0].set_ylim(minAOD,maxAOD)
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
# ax[0,0].scatter(true, rtrv, '.',  c=clrVar, markersize=MS, alpha=pointAlpha)
ax[0,0].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[0,0].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(115, -147), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[0,0].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)


# Modal AOD
# tauWght = lambda tau,var: np.sum(var*tau)/np.sum(tau)
# true = np.asarray([rf['aodMode'][1][waveInd] for rf in simBase.rsltFwd])[keepInd]
# rtrv = np.asarray([rb['aodMode'][1][waveInd] for rb in simBase.rsltBck])[keepInd]
# minAOD = np.min(true)*0.95
# maxAOD = np.max(true)*1.05
# ax[0,1].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
# ax[0,1].set_title('Asym. Param.')
# ax[0,1].set_xlabel(xlabel)
# ax[0,1].set_xlim(minAOD,maxAOD)
# ax[0,1].set_ylim(minAOD,maxAOD)
# ax[0,1].set_xticks(np.arange(minAOD, maxAOD, 500))
# ax[0,1].set_yticks(np.arange(minAOD, maxAOD, 500))
# ax[0,1].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
# Rcoef = np.corrcoef(true, rtrv)[0,1]
# RMSE = np.sqrt(np.median((true - rtrv)**2))
# bias = np.mean((rtrv-true))
# textstr = frmt % (Rcoef, RMSE, bias)
# tHnd = ax[0,1].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
#                     textcoords='offset points', color=clrText, fontsize=FS)
#
"""
# HEIGHT
tauWght = lambda tau,var: np.sum(var*tau)/np.sum(tau)
true = np.asarray([tauWght(rf['aodMode'][:,waveInd], rf['height']) for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([tauWght(rf['aodMode'][:,waveInd], rf['height']) for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
ax[0,1].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[0,1].set_title('Layer Height')
ax[0,1].set_xlabel(xlabel)
ax[0,1].set_xlim(minAOD,maxAOD)
ax[0,1].set_ylim(minAOD,maxAOD)
ax[0,1].set_xticks(np.arange(minAOD, maxAOD, 500))
ax[0,1].set_yticks(np.arange(minAOD, maxAOD, 500))
ax[0,1].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[0,1].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)


"""

# ANGSTROM
aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltFwd])[keepInd]
logLamdRatio = np.log(simBase.rsltFwd[0]['lambda'][waveInd]/simBase.rsltFwd[0]['lambda'][waveInd2])
true = -np.log(true/aod2)/logLamdRatio
aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltBck])[keepInd]
rtrv = -np.log(rtrv/aod2)/logLamdRatio
minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
ax[0,1].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[0,1].set_title('Angstrom Exponent')
ax[0,1].set_xlabel(xlabel)
ax[0,1].set_xlim(minAOD,maxAOD)
ax[0,1].set_ylim(minAOD,maxAOD)
ax[0,1].set_xticks(np.arange(minAOD, maxAOD, 0.5))
ax[0,1].set_yticks(np.arange(minAOD, maxAOD, 0.5))
ax[0,1].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[0,1].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)


# SSA
true = np.asarray([rf['ssa'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['ssa'][waveInd] for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.95
# minAOD = 0.735
maxAOD = 1
ax[0,2].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[0,2].set_title('SSA')
ax[0,2].set_xlabel(xlabel)
ax[0,2].set_xticks(np.arange(0.75, 1.01, 0.05))
ax[0,2].set_yticks(np.arange(0.75, 1.01, 0.05))
ax[0,2].set_xlim(minAOD,maxAOD)
ax[0,2].set_ylim(minAOD,maxAOD)
ax[0,2].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[0,2].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)


# g
true = np.asarray([rf['g'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['g'][waveInd] for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
ax[1,0].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[1,0].set_title('g')
ax[1,0].set_xlabel(xlabel)
ax[1,0].set_ylabel('Retrieved')
ax[1,0].set_xlim(minAOD,maxAOD)
ax[1,0].set_ylim(minAOD,maxAOD)
ax[1,0].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[1,0].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(115, -147), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[1,0].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)


# rEff Fine
true = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
ax[1,1].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[1,1].set_title('r_eff Fine')
ax[1,1].set_xlabel(xlabel)
ax[1,1].set_ylabel('Retrieved')
ax[1,1].set_xlim(minAOD,maxAOD)
ax[1,1].set_ylim(minAOD,maxAOD)
ax[1,1].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[1,1].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(115, -147), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[1,1].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

# rEff Coarse
true = np.asarray([rf['rEffMode'][1] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['rEffMode'][1] for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
ax[1,2].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[1,2].set_title('r_eff Coarse')
ax[1,2].set_xlabel(xlabel)
ax[1,2].set_ylabel('Retrieved')
ax[1,2].set_xlim(minAOD,maxAOD)
ax[1,2].set_ylim(minAOD,maxAOD)
ax[1,2].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[1,2].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(115, -147), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[1,2].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

plt.tight_layout()
plt.ion()
plt.show()
