#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 17:21:42 2019

@author: wrespino
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import copy
import pylab
from simulateRetrieval import simulation
import miscFunctions as mf

#simB.analyzeSim()
#instruments = ['img01visnir','img02visnir','lidar09+img02visnir','lidar05+img02visnir'] #1
instruments = ['polar07'] #1
#conCases = ['marine', 'pollution','smoke','marine+pollution','marine+smoke','Smoke+pollution'] #6
#conCases = ['variablefinenonsph','variablefinenonsph','variablefinenonsph','variablefinenonsph'] #6
#conCases = ['case02a','case02b','case02c','case03','case07']
conCases = []
for caseLet in ['a','b','c','d','e','f']:
#    conCases.append('case06'+caseLet)
    conCases.append('case06'+caseLet+'monomode')
#    if caseLet in ['e','f']:
#        conCases.append('case06'+caseLet+'nonsph')
#        conCases.append('case06'+caseLet+'monomode'+'nonsph') #21 total
#conCases = ['variable']
SZAs = [0, 30, 60] # 3
SZAs = [0]
Phis = [0] # 1 -> N=18 Nodes
#tauVals = [0.3, 1.0, 3.0] #
tauVal = 1.0
#tauVals = [0.04, 0.08, 0.12, 0.18, 0.35] # NEED TO MAKE THIS CHANGE FILE NAME
N = len(SZAs)*len(conCases)*len(Phis)*len(instruments)
#tauVals = [0.18] # NEED TO MAKE THIS CHANGE FILE NAME
gridPlots = False
#l = 3
#lVals = [0,3,5,10]
lVals = [3]

tag = 'Figure'

totBiasVars = ['aod', 'ssa', 'g','aodMode','n','rEffCalc'] # if it is a multi dim array we take first index (aodmode and n)
#totVars = ['aod', 'ssa', 'rEffCalc','g','height']
totVars = ['aod', 'ssa', 'g']
#modVars = ['aodMode', 'ssaMode']  # reff should be profile
modVars = ['n', 'aodMode', 'ssaMode']  # reff should be profile
#trgt = {'aod':[0.02], 'ssa':[0.03], 'g':[0.02], 'height':[500], 'rEffCalc':[0.0], 'aodMode':[0.02,0.02], 'ssaMode':[0.03,0.03], 'n':[0.025,0.025,0.025]} # look at total and fine/coarse 
trgt = {'aod':[0.02], 'ssa':[0.03], 'g':[0.02], 'aodMode':[0.02,0.02], 'ssaMode':[0.03,0.03], 'n':[0.025,0.025,0.025]} # look at total and fine/coarse 
#trgt = {'aod':[0.02], 'ssa':[0.03], 'g':[0.02], 'height':[1000], 'rEffCalc':[0.0], 'aodMode':[0.02,0.02], 'ssaMode':[0.03,0.03], 'n':[0.025]} # only look at one mode (code below will work even if RMSE is calculated for fine/coarse too as long as n is listed under totVars)
#trgtRel = {'aod':0.05, 'rEffCalc':0.20, 'aodMode':0.05} # this part must be same for every mode but absolute component above can change
trgtRel = {'aod':0.05, 'aodMode':0.05} # this part must be same for every mode but absolute component above can change
#trgt = {'aod':0.025, 'ssa':0.04, 'aodMode':[0.02,0.02], 'n':[0.025,0.025,0.025], 'ssaMode':[0.05,0.05], 'rEffCalc':0.05}
#aod fine Mode: 0.02+/-0.05AOD (same for total AOD)
# n: 0.025 (total)
# g is also in SATM...
# rEffCalc and it should be 20%
# ssa total 0.03

saveStart = '/Users/wrespino/synced/Working/SIM9_conCasesV22_Pol07/SIM_'
#saveStart = '/Users/wrespino/synced/Working/SIM8/SIM_'

cm = pylab.get_cmap('viridis')

# def getGVlabels(totVars, modVars, trgt)
gvNames = copy.copy(totVars)
for mv in modVars:
    for i,nm in enumerate(['', '_{fine}','_{coarse}']):
        if i>0 or np.size(trgt[mv]) > 2: # HINT: this assumes two modes!
            gvNames.append(mv+nm)
gvNames = ['$'+mv.replace('Mode','').replace('rEffCalc','r_{eff}').replace('ssa', 'SSA').replace('aod','AOD')+'$' for mv in gvNames]

sizeMat = [1,1,1, len(instruments), len(conCases), len(SZAs), len(Phis)]
Nvars = np.hstack([x for x in trgt.values()]).shape[0]
Ntau = len(lVals)
tauLeg = []
totBias = dict([]) # only valid for last of whatever we itterate through on the next line
figC, axC = plt.subplots(figsize=(10,6))
axC.set_prop_cycle('color', plt.cm.Dark2(np.linspace(0,1,6)))
for tauInd, lInd in enumerate(lVals):
    harvest = np.zeros([Nvars, N])
    farmers = []    
    for n in range(N):
        ind = [n//np.prod(sizeMat[i:i+3])%sizeMat[i+3] for i in range(4)]
        paramTple = (instruments[ind[0]], conCases[ind[1]], SZAs[ind[2]], Phis[ind[3]], tauVal)
        savePath = saveStart + '%s_case-%s_sza%d_phi%d_tFct%4.2f_V2.pkl' % paramTple #SIM3_polar07_case-Smoke+pollution_sza30_phi0_tFct0.04_V2.pkl
        farmers.append('%s($θ_s=%d,φ=%d$)' % paramTple[1:4])
        farmers[-1] = farmers[-1].replace('pollution','POLL').replace('smoke','BB').replace('marine','MRN')
        simB = simulation(picklePath=savePath)
        Nsims = len(simB.rsltBck)
#        lInd = l+1 if 'lidar' in instruments[ind[0]] and l>0 else l
        print('---')
        print(farmers[-1])
        print(savePath)
        print('AODf=%4.2f, AODc=%4.2f, Nsim=%d' % (simB.rsltFwd['aodMode'][0,lInd], simB.rsltFwd['aodMode'][1,lInd], Nsims))
        print(instruments[ind[0]])
        print('Spectral variables for λ = %4.2f μm'% simB.rsltFwd['lambda'][lInd])
        try:
#            assert False
            rmse, bias = simB.analyzeSim(lInd, modeCut=None) # HINT: this will break in cases with differnt number of fwd and back modes
        except (ValueError, AssertionError):
            rmse, bias = simB.analyzeSim(lInd, modeCut=0.5) # HINT: this is much slower than the above
        print('---')
        for vr in totBiasVars:
            if n == 0: # allocate the array
                totBias[vr] = np.ones([N*Nsims, bias[vr].shape[1]])
            totBias[vr][n*Nsims:(n+1)*Nsims,:] = bias[vr]
        
        i=0
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
                # PLOT 3: print PDF of AOD as a function of case
        aodDiffRng = 0.08
        kern = st.gaussian_kde(bias['aod'][np.abs(bias['aod'])<aodDiffRng])
        xAxisVals = np.linspace(-aodDiffRng, aodDiffRng, 500)                
        axC.plot(xAxisVals, kern.pdf(xAxisVals), '-.')
    axC.legend([cc[0:7] for cc in conCases])
    if gridPlots: mf.gridPlot(farmers, gvNames, harvest)    
    plt.rcParams.update({'font.size': 14})
    if tauInd==0: 
        figB, axB = plt.subplots(figsize=(4.8,6))
        axB.plot([1,1], [0,5*(Nvars+1)], ':', color=0.65*np.ones(3))
    pos = Ntau*np.r_[0:harvest.shape[0]]+0.7*tauInd
    indGood = [not 'InsertSkipFieldHere' in gn for gn in gvNames]
    hnd = axB.boxplot(harvest[indGood,:].T, vert=0, patch_artist=True, positions=pos[indGood], sym='.')
    if tauInd == 10:
        [hnd['boxes'][i].set_facecolor([0,0,1]) for i in range(len(hnd['boxes']))]
        [hf.set_markeredgecolor([0,0,1]) for hf in hnd['fliers']]
    elif tauInd == 11:
        [hnd['boxes'][i].set_facecolor([1,0,0]) for i in range(len(hnd['boxes']))]
        [hf.set_markeredgecolor([1,0,0]) for hf in hnd['fliers']]
    else:
        [hnd['boxes'][i].set_facecolor(cm((tauInd)/(Ntau))) for i in range(len(hnd['boxes']))]
        [hf.set_markeredgecolor(cm((tauInd)/(Ntau))) for hf in hnd['fliers']]
    tauLeg.append(hnd['boxes'][0])
    
axB.set_xscale('log')
axB.set_xlim([0.05,50])
axB.set_ylim([-0.8, Ntau*harvest.shape[0]])
plt.sca(axB)
plt.yticks(Ntau*(np.r_[0:(harvest.shape[0])]+0.1*Ntau), gvNames)
#lgHnd = axB.legend(tauLeg[::-1], ['τ = %4.2f' % τ for τ in tauVals[::-1]], loc='center left')
#lgHnd.draggable()
axB.yaxis.set_tick_params(length=0)
figB.tight_layout()
figSavePath = '/Users/wrespino/Documents/'+tag+'_case-%s.png' % paramTple[1]
figB.savefig(figSavePath, dpi=600, facecolor='w', edgecolor='w', orientation='portrait', pad_inches=0.1)

# PLOT 4: print PDF as a fuction of variable type
trgt = {'aod':0.03, 'ssa':0.03, 'g':0.02, 'aodMode':0.03, 'n':0.025} # look at total and fine/coarse aod=0.2
figD, axD = plt.subplots(2,3,figsize=(12,7))
for i,vr in enumerate(totBiasVars):
    curAx = axD[i//3,i%3]
    data = totBias[vr][:,0]
    dataCln = data[np.abs(data) < np.percentile(np.abs(data),98)]
    curAx.hist(dataCln,100)
    if vr in trgt:
        t = trgt[vr]
        curAx.plot([-trgt[vr], -trgt[vr]], [curAx.get_ylim()[0], curAx.get_ylim()[1]],'--k')
        curAx.plot([trgt[vr], trgt[vr]], [curAx.get_ylim()[0], curAx.get_ylim()[1]],'--k')
        print('%s - %d%%' % (vr, 100*np.sum(np.abs(dataCln) < trgt[vr])/len(dataCln)))
    curAx.set_yticks([], []) 
figD.tight_layout()    
        
#a = [2,2,1,1,0,0,1,0,0,0]
#S = lambda a,σ: 5*np.sum(a*(1+σ**2)**(np.log2(4/5)))/np.sum(a)
#print(np.mean([S(a,σ) for σ in harvest.T]))