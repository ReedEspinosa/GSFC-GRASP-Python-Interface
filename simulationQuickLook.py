#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from simulateRetrieval import simulation
from glob import glob

waveInd = 2
fnPtrn = 'Test52_modis_HuamboVegetation_tFct1.000_sza*_phi*_n0_nAng*.pkl'
inDirPath = '/Users/wrespino/synced/Working/TASNPP_simulation00/'
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
        Nrepeats = 1 if NrsltBck==len(simA.rsltFwd) else NrsltBck
        for _ in range(Nrepeats): simBase.rsltFwd = np.r_[simBase.rsltFwd, simA.rsltFwd]
        simBase.rsltBck = np.r_[simBase.rsltBck, simA.rsltBck]
    simBase.saveSim(os.path.join(inDirPath,saveFN))
    print('Saving to %s - %d' % (saveFN, len(simBase.rsltBck)))
else:
    print('No files found')
print('--')    

simOutputPath = os.path.join(inDirPath, saveFN)
sim = simulation(picklePath=simOutputPath)
pprint(sim.analyzeSim(waveInd)[0])

# ind = 90
# plt.figure()
# plt.plot(sim.rsltBck[ind]['lambda'], sim.rsltBck[ind]['fit_I'][0], 'x')
# plt.plot(sim.rsltBck[ind]['lambda'], sim.rsltBck[ind]['meas_I'][0], 'o')
# plt.plot(sim.rsltFwd[ind]['lambda'], sim.rsltFwd[ind]['fit_I'][0])
# plt.legend(['fit','meas','true'])
# 

FS = 12
fig, ax = plt.subplots(1,2, figsize=(10,5))
true = np.asarray([rf['aod'][waveInd] for rf in sim.rsltFwd])
rtrv = np.asarray([rf['aod'][waveInd] for rf in sim.rsltBck])
minSSA = np.min(true)*0.95
ax[0].plot(true, rtrv, '.')
ax[0].plot([minSSA,1], [minSSA,1], 'k')
ax[0].set_xlim(minSSA,1)
ax[0].set_ylim(minSSA,1)
ax[0].set_title('AOD')
ax[0].set_xlabel('True')
ax[0].set_ylabel('Retrieved')
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
textstr = 'N=%d\nR=%.3f\nRMS=%.3f\nbias=%.3f' % (len(true), Rcoef, RMSE, bias)
tHnd = ax[0].annotate(textstr, xy=(0, 1), xytext=(4.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color='k', fontsize=FS)
tHnd.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='None'))

true = np.asarray([rf['ssa'][waveInd] for rf in sim.rsltFwd])
rtrv = np.asarray([rf['ssa'][waveInd] for rf in sim.rsltBck])
minSSA = np.min(true)*0.95
ax[1].plot(true, rtrv, '.')
ax[1].plot([minSSA,1], [minSSA,1], 'k')
ax[1].set_xlim(minSSA,1)
ax[1].set_ylim(minSSA,1)
ax[1].set_title('SSA')
ax[1].set_ylabel('Retrieved')
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.mean((true - rtrv)**2))
bias = np.mean((rtrv-true))
textstr = 'N=%d\nR=%.3f\nRMS=%.3f\nbias=%.3f' % (len(true), Rcoef, RMSE, bias)
tHnd = ax[1].annotate(textstr, xy=(0, 1), xytext=(4.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color='k', fontsize=FS)
tHnd.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='None'))

plt.tight_layout()
plt.ion()
plt.show()


