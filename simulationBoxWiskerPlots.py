#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 17:21:42 2019

@author: wrespino
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
import pylab
from simulateRetrieval import simulation
#simB.analyzeSim()
#instruments = ['img01visnir','img02visnir','lidar09+img02visnir','lidar05+img02visnir'] #1
instruments = ['img01','img02','lidar09+img02','lidar05+img02'] #1
#instruments = [img01','img02', 'lidar09+img02'] #1
#conCases = ['marine', 'pollution','smoke','marine+pollution','marine+smoke','Smoke+pollution'] #6
#conCases = ['variablefinenonsph','variablefinenonsph','variablefinenonsph','variablefinenonsph'] #6
conCases = ['variable'] #6
SZAs = [0, 30, 60] # 3
Phis = [0] # 1 -> N=18 Nodes
N = 15
tauVals = [0.04, 0.08, 0.12, 0.18, 0.35] # NEED TO MAKE THIS CHANGE FILE NAME
#tauVals = [0.18] # NEED TO MAKE THIS CHANGE FILE NAME
gridPlots = False
l = 0
tag = 'Figure'

totVars = ['aod', 'ssa', 'rEffCalc','g','height', 'n']
modVars = ['aodMode', 'ssaMode']  # reff should be profile
#modVars = ['n', 'aodMode', 'ssaMode']  # reff should be profile
#trgt = {'aod':[0.01], 'ssa':[0.01], 'g':[0.02], 'height':[250], 'rEffCalc':[0.0], 'aodMode':[0.01,0.01], 'ssaMode':[0.01,0.01], 'n':[0.02,0.02,0.02]}
trgt = {'aod':[0.02], 'ssa':[0.02], 'g':[0.02], 'height':[1000], 'rEffCalc':[0.0], 'aodMode':[0.02,0.02], 'ssaMode':[0.02,0.02], 'n':[0.02]}
trgtRel = {'aod':0.03, 'rEffCalc':0.10, 'aodMode':0.03} # this part must be same for every mode but absolute component above can change
#trgt = {'aod':0.025, 'ssa':0.04, 'aodMode':[0.02,0.02], 'n':[0.025,0.025,0.025], 'ssaMode':[0.05,0.05], 'rEffCalc':0.05}
#aod fine Mode: 0.02+/-0.05AOD (same for total AOD)
# n: 0.025 (total)
# g is also in SATM...
# rEffCalc and it should be 20%
# ssa total 0.03

saveStart = '/Users/wrespino/synced/Working/SIM8/SIM_oneRI_'

cm = pylab.get_cmap('viridis')

gvNames = copy.copy(totVars)
for mv in modVars:
    for i,nm in enumerate(['', '_{fine}','_{coarse}']):
        if i>0 or np.size(trgt[mv]) > 2: # HINT: this assumes two modes!
            gvNames.append(mv+nm)
gvNames = ['$'+mv.replace('Mode','').replace('rEffCalc','r_{eff}').replace('ssa', 'SSA').replace('aod','AOD')+'$' for mv in gvNames]
sizeMat = [1,1,1, len(tauVals), len(conCases), len(SZAs), len(Phis)]
Nvars = np.hstack([x for x in trgt.values()]).shape[0]
Ntau = len(instruments)
tauLeg = []
for tauInd, instrument in enumerate(instruments):
    harvest = np.zeros([Nvars, N])
    farmers = []    
    for n in range(N):
        ind = [n//np.prod(sizeMat[i:i+3])%sizeMat[i+3] for i in range(4)]
        paramTple = (instrument, conCases[ind[1]], SZAs[ind[2]], Phis[ind[3]], tauVals[ind[0]])
        savePath = saveStart + '%s_case-%s_sza%d_phi%d_tFct%4.2f_V2.pkl' % paramTple #SIM3_polar07_case-Smoke+pollution_sza30_phi0_tFct0.04_V2.pkl
        farmers.append('%s($θ_s=%d,φ=%d$)' % paramTple[1:4])
        farmers[-1] = farmers[-1].replace('pollution','POLL').replace('smoke','BB').replace('marine','MRN')
        simB = simulation(picklePath=savePath)
        lInd = l+1 if 'lidar' in instrument and l>0 else l
        try:
            rmse = simB.analyzeSim(lInd, modeCut=None)[0] # HINT: this will break in cases with differnt number of fwd and back modes
        except ValueError:
            rmse = simB.analyzeSim(lInd, modeCut=0.5)[0] 
        i=0
        print('---')
        print(farmers[-1])
        print(savePath)
        print('AODf=%4.2f, AODc=%4.2f' % (simB.rsltFwd['aodMode'][0,lInd], simB.rsltFwd['aodMode'][1,lInd]))
        print(instrument)
        print('Spectral variables for λ = %4.2f μm'% simB.rsltFwd['lambda'][lInd])
        print('---')
        for vr in totVars+modVars:
            for t,tg in enumerate(trgt[vr]):
                if vr in trgtRel.keys():
                    if np.isscalar(simB.rsltFwd[vr]):
                        true = simB.rsltFwd[vr]
                    elif simB.rsltFwd[vr].ndim==1:
                        true = simB.rsltFwd[vr][lInd]
                    else:
                        true = simB.rsltFwd[vr][t,lInd]
                    harvest[i,n] = (tg+trgtRel[vr]*true)/np.atleast_1d(rmse[vr])[t]
                else:
                    harvest[i,n] = tg/np.atleast_1d(rmse[vr])[t]
                i+=1
    if gridPlots:
        plt.rcParams.update({'font.size': 10})
        fig, ax = plt.subplots(figsize=(15,6), frameon=False)
        im = ax.imshow(np.sqrt(harvest), 'seismic', vmin=0, vmax=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(farmers)))
        ax.set_yticks(np.arange(len(gvNames)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(farmers)
        ax.set_yticklabels(gvNames)
        ax.xaxis.set_tick_params(length=0)
        ax.yaxis.set_tick_params(length=0)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        for i in range(len(gvNames)):
            for j in range(len(farmers)):
                valStr = '%3.1f' % harvest[i, j]
                clr = 'w' if np.abs(harvest[i, j]-1)>0.5 else 'k'
        #        clr = np.min([np.abs(harvest[i, j]-1)**3, 1])*np.ones(3)
                text = ax.text(j, i, valStr,
                               ha="center", va="center", color=clr, fontsize=9)
        fig.tight_layout()
     
    plt.rcParams.update({'font.size': 14})
    if tauInd==0: 
        figB, axB = plt.subplots(figsize=(4.8,6))
        axB.plot([1,1], [0,5*(Nvars+1)], ':', color=0.65*np.ones(3))
    pos = Ntau*np.r_[0:harvest.shape[0]]+0.7*tauInd
    indGood = [not 'InsertSkipFieldHere' in gn for gn in gvNames]
    hnd = axB.boxplot(harvest[indGood,:].T, vert=0, patch_artist=True, positions=pos[indGood], sym='.')
    if tauInd == 0:
        [hnd['boxes'][i].set_facecolor([0,0,1]) for i in range(len(hnd['boxes']))]
        [hf.set_markeredgecolor([0,0,1]) for hf in hnd['fliers']]
    elif tauInd == 1:
        [hnd['boxes'][i].set_facecolor([1,0,0]) for i in range(len(hnd['boxes']))]
        [hf.set_markeredgecolor([1,0,0]) for hf in hnd['fliers']]
    else:
        [hnd['boxes'][i].set_facecolor(cm((tauInd-2)/(Ntau-2))) for i in range(len(hnd['boxes']))]
        [hf.set_markeredgecolor(cm((tauInd-2)/(Ntau-2))) for hf in hnd['fliers']]
    tauLeg.append(hnd['boxes'][0])
axB.set_xscale('log')
axB.set_xlim([0.1,34.5])
axB.set_ylim([-0.8, Ntau*harvest.shape[0]])
plt.sca(axB)
plt.yticks(Ntau*(np.r_[1:(harvest.shape[0]+1)]-0.7), gvNames)
#lgHnd = axB.legend(tauLeg[::-1], ['τ = %4.2f' % τ for τ in tauVals[::-1]], loc='center left')
#lgHnd = axB.legend(tauLeg, ['%s' % τ for τ in instruments], loc='center left')
#lgHnd.draggable()
axB.yaxis.set_tick_params(length=0)
figB.tight_layout()

figSavePath = '/Users/wrespino/Documents/'+tag+'_case-%s.png' % paramTple[1]
figB.savefig(figSavePath, dpi=600, facecolor='w', edgecolor='w', orientation='portrait', pad_inches=0.1)

#a = [2,2,1,1,0,0,1,0,0,0]
#S = lambda a,σ: 5*np.sum(a*(1+σ**2)**(np.log2(4/5)))/np.sum(a)
#print(np.mean([S(a,σ) for σ in harvest.T]))