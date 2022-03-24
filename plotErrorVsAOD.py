#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pprint import pprint
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from simulateRetrieval import simulation
from glob import glob


waveInd = 0
waveInd2 = 5
waveIndAOD = 3
fineIndFwd = [0,2]
fineIndBck = [0]
# pklDataPath = '/Users/wrespino/Synced/Working/OSSE_Test_Run/MERGED_ss450-g5nr.leV210.GRASP.example.polarimeter07.200608ALL_ALLz.pkl' # None to skip reloading of data
# pklDataPath = '/Users/wrespino/Synced/AOS/A-CCP/Assessment_8K_Sept2020/SIM17_SITA_SeptAssessment_AllResults_MERGED/DRS_V01_polar07_caseAll_tFct1.00_orbSS_multiAngles_nAll_nAngALL.pkl' # None to skip reloading of data
pklDataPath = None # None to skip reloading of data
plotSaveDir = '/Users/wrespino/Synced/AOS/PLRA/Figures'
surf2plot = 'ocean' # land, ocean or both
aodMax = 9990.801 # only for plot limits
hist2D = True
Nbins = 300 # NbinsxNbins bins in hist density plots

varVsAOD = False
fnTag = 'AllCases'
scatAlpha = 0.01
scatSize = 7
MS = 1
FS = 10
LW121 = 1
clrText = [0.5,0,0.0]
clrText = [0.0,0,0.5]
errFun = lambda t,r : np.abs(r-t)
aodWght = lambda x,τ : np.sum(x*τ)/np.sum(τ)
version = 'EE_XXX-polar07-SS-G5NR-V12' # for PDF file name
ttlTxt = 'Polarimeter07-only, polar, Nature Run'

if varVsAOD:
    fig, ax = plt.subplots(1,2, figsize=(7.7,3.5))
else:
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax = [ax]
plt.locator_params(nbins=3)


if pklDataPath is not None:
    simBase = simulation(picklePath=pklDataPath)
    print('Loading from %s - %d' % (pklDataPath, len(simBase.rsltBck)))
print('--')

# print general stats to console
print('Showing results for %5.3f μm' % simBase.rsltFwd[0]['lambda'][waveInd])
# MAY NEED FOR Reff?
# pprint(simBase.analyzeSim(waveInd)[0]) 
# simBase._addReffMode(0.7, True) # reframe with cut at 1 micron diameter


if 'land_prct' not in simBase.rsltFwd[0] or surf2plot=='both':
    keepInd = range(len(simBase.rsltFwd))
    if not surf2plot=='both': print('NO land_prct KEY! Including all surface types...')
else:
    lp = np.array([rf['land_prct'] for rf in simBase.rsltFwd])
    keepInd = lp>99 if surf2plot=='land' else lp<1 if surf2plot=='ocean' else lp>-1



# apply convergence filter
# simBase.conerganceFilter(forceχ2Calc=True) # ours looks more normal, but GRASP's produces slightly lower RMSE
costThresh = np.percentile([rb['costVal'] for rb in simBase.rsltBck[keepInd]], 95)
keepInd = np.logical_and(keepInd, [rb['costVal']<costThresh for rb in simBase.rsltBck])

# variable to color point by in all subplots
# clrVar = np.sqrt([rb['costVal'] for rb in simBase.rsltBck[keepInd]]) # this is slow!
clrVar = np.asarray([rb['costVal'] for rb in simBase.rsltBck[keepInd]]) 
print('%d/%d fit surface type %s and convergence filter' % (keepInd.sum(), len(simBase.rsltBck), surf2plot))

logScatPlot=False
fineAOD = False
wavelng = simBase.rsltFwd[0]['lambda'][waveInd]
# AOD
# ylabel = 'AOD (λ=%4.2fμm)' % wavelng
# true = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
# rtrv = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]
# maxVar = 2
# aodMin = 0.0 # does not apply to AOD plot
# # EE_fun = lambda t : 0.02+0.05*t
# # ttlTxt = ttlTxt + ', EE=±0.02+0.05*τ'
# EE_fun = lambda t : 0.05+0.05*t
# ttlTxt = ttlTxt + ', EE=±0.05+0.05*τ'
# AOD Fine
# ylabel = 'AOD_fine (λ=%4.2fμm)' % wavelng
# true = np.asarray([rf['aodMode'][fineIndFwd, waveIndAOD].sum() for rf in simBase.rsltFwd])[keepInd]
# rtrv = np.asarray([rf['aodMode'][fineIndBck, waveIndAOD].sum() for rf in simBase.rsltBck])[keepInd]
# maxVar = None
# aodMin = 0.0 # does not apply to AOD plot
# EE_fun = lambda t : 0.02+0.05*t
# ttlTxt = ttlTxt + ', EE=±0.02+0.05*τ'
# EE_fun = lambda t : 0.05+0.05*t
# ttlTxt = ttlTxt + ', EE=±0.05+0.05*τ'

# AAOD
# ylabel = 'AAOD (λ=%4.2fμm)' % wavelng
# true = np.asarray([(1-rf['ssa'][waveInd])*rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
# rtrv = np.asarray([(1-rb['ssa'][waveInd])*rb['aod'][waveInd] for rb in simBase.rsltBck])[keepInd]
# maxVar = 0.11
# # maxVar = None
# aodMin = 0.0 
# EE_fun = lambda t : np.maximum(0.003, t*0.3)
# ttlTxt = ttlTxt + ', EE=max(0.003, 30%)'

# 1-SSA
# ylabel = 'Coalbedo (λ=%4.2fμm)' % wavelng
# true = np.asarray([(1-rf['ssa'][waveInd]) for rf in simBase.rsltFwd])[keepInd]
# rtrv = np.asarray([(1-rb['ssa'][waveInd]) for rb in simBase.rsltBck])[keepInd]
# maxVar = 0.2
# aodMin = 0.3 # does not apply to AOD plot
# ANGSTROM
# ylabel = 'AE (%4.2f/%4.2f μm)' % (wavelng, simBase.rsltFwd[0]['lambda'][waveInd2])
# aod1 = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
# aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltFwd])[keepInd]
# logLamdRatio = np.log(simBase.rsltFwd[0]['lambda'][waveInd]/simBase.rsltFwd[0]['lambda'][waveInd2])
# true = -np.log(aod1/aod2)/logLamdRatio
# aod1 = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]
# aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltBck])[keepInd]
# rtrv = -np.log(aod1/aod2)/logLamdRatio
# maxVar = 2.25
# aodMin = 0.05 # does not apply to AOD plot
# g
# ylabel = 'Asym. Param. (λ=%4.2fμm)' % wavelng
# true = np.asarray([(rf['g'][waveInd]) for rf in simBase.rsltFwd])[keepInd]
# rtrv = np.asarray([(rb['g'][waveInd]) for rb in simBase.rsltBck])[keepInd]
# maxVar = None
# aodMin = 0.15 # does not apply to AOD plot
# vol FMF
# ylabel = 'Submicron Volum. Frac.'
# def fmfCalc(r,dvdlnr):
#     cutRadius = 0.5
#     fInd = r<=cutRadius
#     logr = np.log(r)
#     return np.trapz(dvdlnr[fInd],logr[fInd])/np.trapz(dvdlnr,logr)
# true = np.asarray([fmfCalc(rf['r'], rf['dVdlnr']) for rf in simBase.rsltFwd])[keepInd]
# rtrv = np.asarray([fmfCalc(rb['r'][0,:], rb['dVdlnr'].sum(axis=0)) for rb in simBase.rsltBck])[keepInd]
# maxVar = None
# aodMin = 0.1 # does not apply to AOD plot
# SPH
# ylabel = 'Volum. Frac. Spherical'
# true = np.asarray([rf['sph'] for rf in simBase.rsltFwd])[keepInd]
# rtrv = np.asarray([aodWght(rf['sph'], rf['vol']) for rf in simBase.rsltBck])[keepInd]
# maxVar = None
# aodMin = 0.1 # does not apply to AOD plot
# k
# ylabel = 'AOD Wghtd. IRI (λ=%4.2fμm)' % wavelng
# true = np.asarray([rf['k'][waveInd] for rf in simBase.rsltFwd])[keepInd]
# rtrv = np.asarray([aodWght(rf['k'][:,waveInd], rf['aodMode'][:,waveInd]) for rf in simBase.rsltBck])[keepInd]
# maxVar = None
# aodMin = 0.2 # does not apply to AOD plot
# logScatPlot=False # This is weird with 2d hist binning
# n
# ylabel = 'AOD Wghtd. RRI (λ=%4.2fμm)' % wavelng
# true = np.asarray([rf['n'][waveInd] for rf in simBase.rsltFwd])[keepInd]
# rtrv = np.asarray([aodWght(rf['n'][:,waveInd], rf['aodMode'][:,waveInd]) for rf in simBase.rsltBck])[keepInd]
# maxVar = None
# aodMin = 0.1 # does not apply to AOD plot
# reff
ylabel = 'Fine Mode Effective Radius'
# simBase._addReffMode(0.5, True) # reframe so pretty much all of the PSD is in the second "coarse" mode
true = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltBck])[keepInd]-0.04
maxVar = 0.32
rtrv[rtrv>maxVar] = maxVar+0.01
aodMin = 0.15  # will be for fine mode given fineAOD=True below
EE_fun = lambda t : 0.04
ttlTxt = ttlTxt + ', EE=0.04μm'
# fineAOD = True
if fineAOD:
    trueAOD = np.asarray([rf['aodMode'][fineIndFwd, waveIndAOD].sum() for rf in simBase.rsltFwd])[keepInd]
else:
    trueAOD = np.asarray([rf['aod'][waveIndAOD] for rf in simBase.rsltFwd])[keepInd]


cmap = plt.cm.YlOrBr
cmap = plt.cm.Reds
# Scatter Plot
vldI = np.logical_and(trueAOD>=aodMin, trueAOD<=aodMax)
if maxVar is None:
    findMaxVar = True
    minVar = np.min(true[vldI])*0.9
    maxVar = np.max(true[vldI])*1.1
else:
    findMaxVar = False
    minVar = 0
ax[0].plot([minVar,maxVar], [minVar,maxVar], 'k', linewidth=LW121)
EE_color = [0.5,0.5,0.5]
ax[0].plot([minVar,maxVar], [minVar-EE_fun(minVar),maxVar-EE_fun(maxVar)], '--', color=EE_color, linewidth=LW121)
ax[0].plot([minVar,maxVar], [minVar+EE_fun(minVar),maxVar+EE_fun(maxVar)], '--', color=EE_color, linewidth=LW121)
if hist2D:
    cnt = ax[0].hist2d(true[vldI], rtrv[vldI], (Nbins,Nbins), norm=mpl.colors.LogNorm(), cmap=cmap)
    cnt[3].set_edgecolor("face")
else:
    ax[0].scatter(true[vldI], rtrv[vldI], c='r', s=scatSize, alpha=scatAlpha)
ax[0].set_xlabel('True %s' % ylabel)
ax[0].set_ylabel('Retrieved %s' % ylabel)
if logScatPlot:
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
ax[0].set_xlim(minVar,maxVar)
ax[0].set_ylim(minVar,maxVar)

Rcoef = np.corrcoef(true[vldI], rtrv[vldI])[0,1]
diffs = rtrv[vldI] - true[vldI]
RMSE = np.sqrt(np.mean(diffs**2))
bias = np.mean(diffs)
inEE = np.sum(np.abs(diffs) < (EE_fun(true[vldI])))/sum(vldI)*100
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f\nIn EE=%%%4.1f'
aodStr = 'AOD_fine' if fineAOD else 'AOD'
tHnd = ax[0].annotate('N=%4d\n(%s>%g)' % (sum(vldI), aodStr, aodMin), xy=(0, 1), xytext=(200, -260), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias, inEE)
tHnd = ax[0].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

# Variable Vs AOD
if varVsAOD:
    vldI = trueAOD<=aodMax
    # rtrv = true + np.random.normal(0,0.1,len(true))
    err = errFun(true[vldI], rtrv[vldI])
    Ndx = 10
    dx = aodMax/round(Ndx)
    aodsX = np.r_[0:aodMax:dx]
    aodsX = np.linspace(0, aodMax - dx, round(Ndx))
    rmseOfX = [(np.sqrt(np.mean(err[np.logical_and(trueAOD[vldI]>x, trueAOD[vldI]<(x+dx))]**2))) for x in aodsX]
    # rmseOfX = [np.percentile(err[np.logical_and(trueAOD[vldI]>x, trueAOD[vldI]<(x+dx))], 84) for x in aodsX]
    if findMaxVar:
        minVar = np.min(err)
        maxVar = np.max(err)
    else:
        minVar = 0
    ax[1].set_ylabel('Error in ' + ylabel)
    ax[1].set_xlabel('True %s (λ=%4.2fμm)' % (aodStr, simBase.rsltFwd[0]['lambda'][waveIndAOD]))
    cnt = ax[1].hist2d(trueAOD[vldI], err, (Nbins,Nbins), norm=mpl.colors.LogNorm(), cmap=cmap)
    cnt[3].set_edgecolor("face")
    # ax[0].plot(aodsX,rmseOfX, '-', color='m', alpha=0.99997, linewidth=2)
    ax[1].plot([aodMin,aodMin],[minVar, maxVar], '--', color='k', alpha=0.4, linewidth=2)
    ax[1].set_xlim(0, aodMax)
    ax[1].set_xticks(np.r_[0:aodMax:0.2])
    ax[1].set_ylim(minVar, maxVar)
    tHnd = ax[1].annotate('N=%4d\n(All AODs)' % sum(vldI), xy=(0, 1), xytext=(90, -4.5), va='top', xycoords='axes fraction',
                textcoords='offset points', color=clrText, fontsize=FS)

plt.title(ttlTxt, fontsize=10)
plt.tight_layout()

ylabelcln = 'AE' if 'AE' in ylabel else ylabel[0:-11]
fn = 'ScatterPlots_AODgt%05.3f_%s_%dnm_%s.pdf' % (aodMin, ylabelcln, (wavelng*1000), version)
fig.savefig(os.path.join(plotSaveDir, fn))
print('Plot saved as %s' % fn)
plt.ion()
plt.show()
