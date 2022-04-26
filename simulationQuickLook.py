#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from simulateRetrieval import simulation
from glob import glob

waveInd = 1
waveInd2 = 3
fnPtrnList = []
#fnPtrn = 'ss450-g5nr.leV210.GRASP.example.polarimeter07.200608*_*z.pkl'
fnPtrn = 'harp02_2modes_AOD_*_550nm*.pkl'
# fnPtrn = 'ss450-g5nr.leV210.GRASP.example.polarimeter07.200608*_1000z.pkl'
inDirPath = '/Users/aputhukkudy/Working_Data/ACCDAM/2022/Campex_Simulations/Mar2022/'\
            'All_Flights/Spherical/Linear/2modes/'
surf2plot = 'ocean' # land, ocean or both
aodMin = 0.1 # does not apply to first AOD plot

fnTag = 'AllCases'
xlabel = 'Simulated Truth'
MS = 2
FS = 10
LW121 = 1
pointAlpha = 0.30
clrText = [0.5,0,0.0]
fig, ax = plt.subplots(2,5, figsize=(15,6))
plt.locator_params(nbins=3)
lightSave = True # Omit PM elements and extinction profiles from MERGED files to save space

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
        if lightSave:
            for pmStr in ['angle', 'p11','p12','p22','p33','p34','p44','range','βext']:
                [rb.pop(pmStr, None) for rb in simA.rsltBck]
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
# lp = np.array([rf['land_prct'] for rf in simBase.rsltFwd])
lp = np.array([0 for rf in simBase.rsltFwd])
keepInd = lp>99 if surf2plot=='land' else lp<1 if surf2plot=='ocean' else lp>-1

# apply convergence filter
# simBase.conerganceFilter(forceχ2Calc=True) # ours looks more normal, but GRASP's produces slightly lower RMSE
costThresh = np.percentile([rb['costVal'] for rb in simBase.rsltBck[keepInd]], 90)
keepInd = np.logical_and(keepInd, [rb['costVal']<costThresh for rb in simBase.rsltBck])

# variable to color point by in all subplots
# clrVar = np.sqrt([rb['costVal'] for rb in simBase.rsltBck[keepInd]]) # this is slow!
clrVar = np.asarray([rb['costVal'] for rb in simBase.rsltBck[keepInd]]) 
print('%d/%d fit surface type %s and convergence filter' % (keepInd.sum(), len(simBase.rsltBck), surf2plot))

# AOD
true = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
ax[0,0].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[0,0].set_title('AOD')
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
tHnd = ax[0,0].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[0,0].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

# AAOD
true = np.asarray([(1-rf['ssa'][waveInd])*rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([(1-rb['ssa'][waveInd])*rb['aod'][waveInd] for rb in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.95
# minAOD = 0.735
maxAOD = 0.15
ax[0,2].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
# ax[0,2].set_title('Co-Albedo (1-SSA)')
ax[0,2].set_title('AAOD')
# ax[0,2].set_xticks(np.arange(0.75, 1.01, 0.05))
# ax[0,2].set_yticks(np.arange(0.75, 1.01, 0.05))
ax[0,2].set_xlim(minAOD,maxAOD)
ax[0,2].set_ylim(minAOD,maxAOD)
ax[0,2].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
tHnd = ax[0,2].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[0,2].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)


# apply AOD min after we plot AOD
keepInd = np.logical_and(keepInd, [rf['aod'][waveInd]>=aodMin for rf in simBase.rsltFwd])
print('%d/%d fit surface type %s and aod≥%4.2f' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, aodMin))
# clrVar = np.sqrt([rb['costVal'] for rb in simBase.rsltBck[keepInd]]) # this is slow!
clrVar = np.asarray([rb['costVal'] for rb in simBase.rsltBck[keepInd]]) 


# apply Reff min
# simBase._addReffMode(0.008, True) # reframe so pretty much all of the PSD is in the second "coarse" mode
simBase._addReffMode(0.7, True) # reframe with cut at 1 micron diameter
#keepInd = np.logical_and(keepInd, [rf['rEffMode']>=2.0 for rf in simBase.rsltBck])
#print('%d/%d fit surface type %s and aod≥%4.2f AND retrieved Reff>2.0μm' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, aodMin))
#clrVar = np.sqrt([rb['rEff']/rf['rEff']-1 for rb,rf in zip(simBase.rsltBck[keepInd], simBase.rsltFwd[keepInd])])

# ANGSTROM
aod1 = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltFwd])[keepInd]
logLamdRatio = np.log(simBase.rsltFwd[0]['lambda'][waveInd]/simBase.rsltFwd[0]['lambda'][waveInd2])
true = -np.log(aod1/aod2)/logLamdRatio
aod1 = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]
aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltBck])[keepInd]
rtrv = -np.log(aod1/aod2)/logLamdRatio
minAOD = np.percentile(true,1) # BUG: Why is Angstrom >50 in at least one OSSE cases?
maxAOD = np.percentile(true,99)
ax[0,1].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[0,1].set_title('Angstrom Exponent')
ax[0,1].set_xlim(minAOD,maxAOD)
ax[0,1].set_ylim(minAOD,maxAOD)
ax[0,1].set_xticks(np.arange(minAOD, maxAOD, 0.5))
ax[0,1].set_yticks(np.arange(minAOD, maxAOD, 0.5))
ax[0,1].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
tHnd = ax[0,1].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[0,1].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

# k
aodWght = lambda x,τ : np.sum(x*τ)/np.sum(τ)
true = np.asarray([rf['k'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([aodWght(rf['k'][:,waveInd], rf['aodMode'][:,waveInd]) for rf in simBase.rsltBck])[keepInd]
# rtrv = 1-np.asarray([rf['ssa'][waveInd] for rf in simBase.rsltFwd])[keepInd]
# true = 1-np.asarray([rb['ssa'][waveInd] for rb in simBase.rsltBck])[keepInd]
# minAOD = np.min(true)*0.95
minAOD = 0.0005
maxAOD = np.max(true)*1.05
ax[0,3].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[0,3].set_title('k')
ax[0,3].set_xscale('log')
ax[0,3].set_yscale('log')
ax[0,3].set_xlim(minAOD,maxAOD)
ax[0,3].set_ylim(minAOD,maxAOD)
ax[0,3].set_xticks([0.001, 0.01])
ax[0,3].set_yticks([0.001, 0.01])
ax[0,3].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[0,3].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[0,3].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

# FMF (vol)
def fmfCalc(r,dvdlnr):
    cutRadius = 0.5
    fInd = r<=cutRadius
    logr = np.log(r)
    return np.trapz(dvdlnr[fInd],logr[fInd])/np.trapz(dvdlnr,logr)
try:
    true = np.asarray([fmfCalc(rf['r'], rf['dVdlnr']) for rf in simBase.rsltFwd])[keepInd]
    rtrv = np.asarray([fmfCalc(rb['r'][0,:], rb['dVdlnr'].sum(axis=0)) for rb in simBase.rsltBck])[keepInd]
    minAOD = 0.01
    maxAOD = 1.0
    ax[0,4].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
    ax[0,4].set_title('Volume FMF')
    ax[0,4].set_xscale('log')
    ax[0,4].set_yscale('log')
    ax[0,4].set_xlim(minAOD,maxAOD)
    ax[0,4].set_ylim(minAOD,maxAOD)
    ax[0,4].set_xticks([minAOD, maxAOD])
    ax[0,4].set_yticks([minAOD, maxAOD])
    ax[0,4].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
    tHnd = ax[0,4].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
                textcoords='offset points', color=clrText, fontsize=FS)
    textstr = frmt % (Rcoef, RMSE, bias)
    tHnd = ax[0,4].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
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
    tHnd = ax[1,0].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
                textcoords='offset points', color=clrText, fontsize=FS)
    textstr = frmt % (Rcoef, RMSE, bias)
    tHnd = ax[1,0].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                        textcoords='offset points', color=clrText, fontsize=FS)
except Exception as err:
    print('Error in plotting FMF: \n error: %s' %err)
    
    # try plotting bland altman
    true = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
    rtrv = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]
    rtrv = true - rtrv
    minAOD = np.min(true)*0.9
    maxAOD = np.max(true)*1.1
    ax[1,0].plot([minAOD,maxAOD], [0,0], 'k', linewidth=LW121)
    ax[1,0].set_title('difference in AOD')
    ax[1,0].set_xlabel(xlabel)
    ax[1,0].set_ylabel('true-retrieved')
    ax[1,0].set_xlim(minAOD,maxAOD)
    ax[1,0].set_ylim(-maxAOD/10,maxAOD/10)
    # ax[1,0].set_yscale('log')
    ax[1,0].set_xscale('log')
    ax[1,0].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    # frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
    tHnd = ax[1,0].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
                textcoords='offset points', color=clrText, fontsize=FS)
    # textstr = frmt % (Rcoef, RMSE, bias)
    # tHnd = ax[1,0].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
    #                     textcoords='offset points', color=clrText, fontsize=FS)    


# sph
true = np.asarray([rf['sph'] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([aodWght(rf['sph'], rf['vol']) for rf in simBase.rsltBck])[keepInd]
minAOD = 0
maxAOD = 100.1
ax[1,1].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[1,1].set_title('spherical vol. frac.')
ax[1,1].set_xlabel(xlabel)
ax[1,1].set_xticks(np.arange(minAOD, maxAOD, 25))
ax[1,1].set_yticks(np.arange(minAOD, maxAOD, 25))
ax[1,1].set_xlim(minAOD,maxAOD)
ax[1,1].set_ylim(minAOD,maxAOD)
ax[1,1].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[1,1].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[1,1].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

# rEff
#simBase._addReffMode(0.008, True) # reframe so pretty much all of the PSD is in the second "coarse" mode
true = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
ax[1,2].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
# ax[1,2].set_title('r_eff Total')
ax[1,2].set_title('Submicron r_eff')
ax[1,2].set_xlabel(xlabel)
ax[1,2].set_xlim(minAOD,maxAOD)
ax[1,2].set_ylim(minAOD,maxAOD)
ax[1,2].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[1,2].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[1,2].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

ttlStr = '%s (λ=%5.3fμm, %s surface, AOD≥%4.2f)' % (saveFN, simBase.rsltFwd[0]['lambda'][waveInd], surf2plot, aodMin)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(ttlStr.replace('MERGED_',''))

# n
true = np.asarray([rf['n'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([aodWght(rf['n'][:,waveInd], rf['aodMode'][:,waveInd]) for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(true)
maxAOD = np.max(true)
ax[1,3].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[1,3].set_title('n')
ax[1,3].set_xlabel(xlabel)
ax[1,3].set_xlim(minAOD,maxAOD)
ax[1,3].set_ylim(minAOD,maxAOD)
ax[1,3].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[1,3].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[1,3].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

# intensity
true = np.sum([rb['meas_I'][:,waveInd] for rb in simBase.rsltBck[keepInd]], axis=1)
rtrv = np.sum([rb['fit_I'][:,waveInd] for rb in simBase.rsltBck[keepInd]], axis=1)
minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
ax[1,4].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[1,4].set_title('sum(intensity)')
ax[1,4].set_xlabel(xlabel)
ax[1,4].set_xlim(minAOD,maxAOD)
ax[1,4].set_ylim(minAOD,maxAOD)
ax[1,4].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[1,4].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[1,4].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)


figSavePath = saveFN.replace('.pkl',('_%s_%s_%04dnm.png' % (surf2plot, fnTag, simBase.rsltFwd[0]['lambda'][waveInd]*1000)))
print('Saving figure to: %s' % figSavePath)
# plt.savefig('' + figSavePath)
#plt.show()

