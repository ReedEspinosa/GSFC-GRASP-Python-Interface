#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
import pickle
import os
import runGRASP as rg
import miscFunctions as ms

class simulation(object):
    def __init__(self, nowPix=None, picklePath=None):
        assert not (nowPix and picklePath), 'Either nowPix or picklePath should be provided, but not both.'
        if picklePath: self.loadSim(picklePath)
        if nowPix is None: return
        assert np.all([np.all(np.diff(mv['meas_type'])>0) for mv in nowPix.measVals]), 'nowPix.measVals[l][\'meas_type\'] must be in ascending order at each l'
        self.nowPix = nowPix
        self.nbvm = np.array([mv['nbvm'] for mv in nowPix.measVals])
        self.rsltBck = None
        self.rsltFwd = None

    def runSim(self, fwdModelYAMLpath, bckYAMLpath, Nsims=100, maxCPU=4, binPathGRASP=None, savePath=None, lightSave=False, intrnlFileGRASP=None, releaseYAML=True, rndIntialGuess=False):
        assert not self.nowPix is None, 'A dummy pixel (nowPix) and error function (addError) are needed in order to run the simulation.' 
        # RUN THE FOWARD MODEL
        gObjFwd = rg.graspRun(fwdModelYAMLpath)
        gObjFwd.addPix(self.nowPix)
        gObjFwd.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=intrnlFileGRASP)
        self.rsltFwd = gObjFwd.readOutput()[0]
        # ADD NOISE AND PERFORM RETRIEVALS
        gObjBck = rg.graspRun(bckYAMLpath, releaseYAML=releaseYAML, quietStart=True) # quietStart=True -> we won't see path of temp, pre-gDB graspRun
        for i in range(Nsims):
            for l, msDct in enumerate(self.nowPix.measVals):
                edgInd = np.r_[0, np.cumsum(self.nbvm[l])]
                msDct['measurements'] = copy.copy(msDct['measurements']) # we are about to write to this
                msDct['measurements'] = msDct['errorModel'](l, self.rsltFwd, edgInd)
                msDct['measurements'][np.abs(msDct['measurements'])<1e-10] = 1e-10 # TODO clean this up, can change sign, not flexible, etc.
            self.nowPix.dtNm = copy.copy(self.nowPix.dtNm+1) # otherwise GRASP will whine
            gObjBck.addPix(self.nowPix)
        gDB = rg.graspDB(gObjBck, maxCPU)
        self.rsltBck = gDB.processData(maxCPU, binPathGRASP, krnlPathGRASP=intrnlFileGRASP, rndGuess=rndIntialGuess)
        # SAVE RESULTS
        if savePath:
            if not os.path.exists(os.path.dirname(savePath)):
                print('savePath (%s) did not exist, creating it...')
                os.makedirs(os.path.dirname(savePath))
            if lightSave:
                for pmStr in ['p11','p12','p22','p33','p34','p44']:
                    [rb.pop(pmStr, None) for rb in self.rsltBck]
            with open(savePath, 'wb') as f:
                pickle.dump(self.rsltBck.tolist()+[self.rsltFwd], f, pickle.HIGHEST_PROTOCOL)

    def loadSim(self, picklePath):
        gDB = rg.graspDB()
        loadData = gDB.loadResults(picklePath)
        self.rsltBck = loadData[:-1]
        self.rsltFwd = loadData[-1]

    def analyzeSim(self, wvlnthInd=0, modeCut=0.5, swapFwdModes = False): # modeCut is fine/coarse seperation radius in um; NOTE: only applies if fwd & inv models have differnt number of modes
        varsSpctrl = ['aod', 'aodMode', 'n', 'k', 'ssa', 'ssaMode', 'g', 'LidarRatio']
        varsMorph = ['rv', 'sigma', 'sph', 'rEffCalc', 'height']
        varsAodAvg = ['n', 'k'] # modal variables for which we will append a aod weighted average RMSE and BIAS value at the FIRST element (expected to be spectral quantity)
        varsSpctrl = [z for z in varsSpctrl if z in self.rsltFwd] # check that the variable is used in current configuration
        varsMorph = [z for z in varsMorph if z in self.rsltFwd]
        varsAodAvg = [z for z in varsAodAvg if z in self.rsltFwd]
        rmsErr = dict()
        bias = dict()
        assert (not self.rsltBck is None) and self.rsltFwd, 'You must call loadSim() or runSim() before you can calculate statistics!'
        for av in varsSpctrl+varsMorph+['rEffMode']:
            if av in varsSpctrl:
                rtrvd = np.array([rs[av][...,wvlnthInd] for rs in self.rsltBck]) # wavelength is always the last dimension
                true = self.rsltFwd[av][...,wvlnthInd]
                if true.ndim==1 and true.shape[0]==2 and swapFwdModes:
                    true = true[::-1]
            elif av in varsMorph:
                rtrvd = np.array([rs[av] for rs in self.rsltBck])
                true = self.rsltFwd[av]
                if true.ndim==1 and true.shape[0]==2 and swapFwdModes:
                    true = true[::-1]
            elif av=='rEffMode':
                true = self.ReffMode(self.rsltFwd,modeCut)
                rtrvd = np.array([self.ReffMode(rs,modeCut) for rs in self.rsltBck])
            if rtrvd.ndim>1 and not av=='rEffMode': # we will seperate fine and coarse modes here
                if modeCut:
                    for i,rs in enumerate(self.rsltBck): 
                        rtrvd[i]= self.volWghtedAvg(rtrvd[i], rs['rv'], rs['sigma'], rs['vol'], modeCut)                
                    true = self.volWghtedAvg(true, self.rsltFwd['rv'], self.rsltFwd['sigma'], self.rsltFwd['vol'], modeCut)
                else:
                    assert len(self.rsltBck[0]['rv'])==len(self.rsltFwd['rv']), 'If modeCut==None, foward and inverted data must have the same number of modes.'
                if av in varsAodAvg: # we also want to calculate the total, mode AOD weighted value of the variable
                    avgVal = [np.sum(rs[av][:,wvlnthInd]*rs['aodMode'][:,wvlnthInd])/np.sum(rs['aodMode'][:,wvlnthInd]) for rs in self.rsltBck]
                    rtrvd = np.vstack([avgVal, rtrvd.T]).T
                    avgVal = np.sum(self.rsltFwd[av][:,wvlnthInd]*self.rsltFwd['aodMode'][:,wvlnthInd])/np.sum(self.rsltFwd['aodMode'][:,wvlnthInd])
                    true = np.r_[avgVal, true]
            if av in ['n','k','sph'] and true.ndim==1 and rtrvd.ndim==1 and true.shape[0]!=rtrvd.shape[0]: # HACK (kinda): if we only retrieve one mode but simulate more we won't report anything
                true = np.mean(true)
            stdDevCut = 2
            if rtrvd.ndim==1:
                rtrvdCln = rtrvd[abs(rtrvd - np.mean(rtrvd)) < stdDevCut * np.std(rtrvd)]
                rmsErr[av] = np.sqrt(np.mean((true-rtrvdCln)**2, axis=0))
            else:
                rmsErr[av] = np.empty(rtrvd.shape[1])
                for m in range(rtrvd.shape[1]):
                    rtrvdCln = rtrvd[:,m][abs(rtrvd[:,m] - np.mean(rtrvd[:,m])) < stdDevCut * np.std(rtrvd[:,m])]                        
                    rmsErr[av][m] = np.sqrt(np.mean((true[m]-rtrvdCln)**2, axis=0))
#            rmsErr[av] = np.sqrt(np.median((true-rtrvd)**2, axis=0))
            bias[av] = rtrvd-true if rtrvd.ndim > 1 else np.atleast_2d(rtrvd-true).T
        return rmsErr, bias
    
    def volWghtedAvg(self, val, rv, sigma, vol, modeCut):
        N = 1e3
        lower = [modeCut/50, modeCut]
        upper = [modeCut, modeCut*50]
        crsWght = []
        for upr, lwr, in zip(upper,lower):
            r = np.logspace(np.log10(lwr),np.log10(upr),N)
            crsWght.append([np.trapz(ms.logNormal(mu, σ, r)[0],r) for mu,σ in zip (rv,sigma)])
        crsWght = np.array(crsWght)*vol
        if val is None: # we just want volume (or area if vol arugment contains area) concentration of each mode
            return np.sum(crsWght,axis=1)
        else: 
            return np.sum(crsWght*val,axis=1)/np.sum(crsWght,axis=1)

    def ReffMode(self, rs, modeCut):
        if not modeCut: 
            dndr = rs['rv']/np.exp(3*rs['sigma']**2)
            return dndr*np.exp(5/2*rs['sigma']**2) # eq. 60 from Grainger's "Useful Formulae for Aerosol Size Distributions"
        Vfc = self.volWghtedAvg(None, rs['rv'], rs['sigma'], rs['vol'], modeCut)
        Amode = rs['vol']/rs['rv']*np.exp(rs['sigma']**2/2) # convert N to rv and then ratio 1st and 4th rows of Table 1 of Grainger's "Useful Formulae for Aerosol Size Distributions"
        Afc = self.volWghtedAvg(None, rs['rv'], rs['sigma'], Amode, modeCut)
        return Vfc/Afc # NOTE: ostensibly this should be Vfc/Afc/3 but we ommited the factor of 3 from Amode
    
    
        
        
        
