#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
import pickle
import os
import datetime as dt
import runGRASP as rg
import miscFunctions as ms

class simulation(object):
    def __init__(self, nowPix=None, picklePath=None):
        assert not (nowPix and picklePath), 'Either nowPix or picklePath should be provided, but not both.'
        if picklePath: self.loadSim(picklePath)
        if nowPix is None: return
        assert np.all([np.all(np.diff(mv['meas_type'])>0) for mv in nowPix.measVals]), 'nowPix.measVals[l][\'meas_type\'] must be in ascending order at each l'
        self.nowPix = copy.deepcopy(nowPix) # we will change this, bad form not to make our own copy
        self.nbvm = np.array([mv['nbvm'] for mv in nowPix.measVals])
        self.rsltBck = None
        self.rsltFwd = None

    def runSim(self, fwdData, bckYAMLpath, Nsims=1, maxCPU=4, binPathGRASP=None, savePath=None, lightSave=False, intrnlFileGRASP=None, releaseYAML=True, rndIntialGuess=False):
        """ <> runs the simulation for given set of simulated and inversion conditions <>
        fwdData -> yml file path for GRASP fwd model OR "results style" list of dicts
        bckYAMLpath -> yml file path for GRASP inversion
        Nsims -> number of noise pertbations applied to fwd model, must be 1 if fwdData is a list of results dicts
        maxCPU -> the retrieval load will be spread accross maxCPU processes
        binPathGRASP -> path to GRASP binary, if None default from graspRun is used
        savePath -> path to save pickle w/ simulated retrieval results, lightSave -> remove PM data
        intrnlFileGRASP -> alternative path to GRASP kernels, overwrites value in YAML files
        releaseYAML=True -> auto adjust back yaml Nλ to match insturment
        rndIntialGuess=True -> overwrite initial guesses in bckYAMLpath w/ uniformly distributed random values between min & max """
        assert not self.nowPix is None, 'A dummy pixel (nowPix) and error function (addError) are needed in order to run the simulation.' 
        # ADAPT fwdData/RUN THE FOWARD MODEL
        if type(fwdData) == str and fwdData[-3:] == 'yml':
            gObjFwd = rg.graspRun(fwdData)
            gObjFwd.addPix(self.nowPix)
            gObjFwd.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=intrnlFileGRASP)
            self.rsltFwd = np.array([gObjFwd.readOutput()[0]])
            loopInd = np.zeros(Nsims, int)
        elif type(fwdData) == list:
            self.rsltFwd = fwdData
            assert Nsims <= 1, 'Running multiple noise perturbations on more than one foward simulation is not supported.' 
            Nsims = 0 # Nsims really has no meaning here so we will use this as a flag
            loopInd = range(len(self.rsltFwd))
        else:
            assert False, 'Unrecognized data type, fwdModelYAMLpath should be path to YAML file or a DICT!'
        # ADD NOISE AND PERFORM RETRIEVALS
        gObjBck = rg.graspRun(bckYAMLpath, releaseYAML=releaseYAML, quietStart=True) # quietStart=True -> we won't see path of temp, pre-gDB graspRun
        for i in loopInd:
            for l, msDct in enumerate(self.nowPix.measVals):
                edgInd = np.r_[0, np.cumsum(self.nbvm[l])]
                msDct['measurements'] = msDct['errorModel'](l, self.rsltFwd[i], edgInd)
                if Nsims == 0:
                    msDct['sza'] = self.rsltFwd[i]['sza'][0,l]
                    msDct['thtv'] = self.rsltFwd[i]['vis'][:,l]
                    msDct['phi'] = self.rsltFwd[i]['fis'][:,l]
                    msDct = self.nowPix.formatMeas(msDct) # this will tile the above msTyp times
            if Nsims == 0:
                self.nowPix.dtObj = self.rsltFwd[i]['datetime'] # ΤΟDO: this produces an integer & only keeps the date part... Should we just ditch this ordinal crap?
                self.nowPix.lat = self.rsltFwd[i]['latitude']
                self.nowPix.lon = self.rsltFwd[i]['longitude']
                if 'land_prct' in self.rsltFwd[i]: self.nowPix.land_prct = self.rsltFwd[i]['land_prct']
            else:
                self.nowPix.dtObj = self.nowPix.dtObj + dt.timedelta(hours=1) # increment hour otherwise GRASP will whine
            gObjBck.addPix(self.nowPix) # addPix performs a deepcopy on nowPix, won't be impact by next iteration through loopInd
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
                    if len(self.rsltFwd) > 1: [rf.pop(pmStr, None) for rf in self.rsltFwd]
            with open(savePath, 'wb') as f:
                pickle.dump(list(self.rsltBck), f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(list(self.rsltFwd), f, pickle.HIGHEST_PROTOCOL)

    def loadSim(self, picklePath):
        with open(picklePath, 'rb') as f:
            self.rsltBck = np.array(pickle.load(f))
            try:
                self.rsltFwd = np.array(pickle.load(f))
            except EOFError: # this was an older file (created before Jan 2020)
                self.rsltFwd = [self.rsltBck[-1]]
                self.rsltBck = self.rsltBck[:-1]
 
    def analyzeSim(self, wvlnthInd=0, modeCut=None, hghtCut=None, swapFwdModes=False): # modeCut is fine/coarse seperation radius in um; NOTE: only applies if fwd & inv models have differnt number of modes
        """ swapFwdModes only works if there are two modes..."""
        if not type(self.rsltFwd) is dict: self.rsltFwd = self.rsltFwd[0] # HACK [VERY BAD] -- remove when we fix this to work with lists 
        assert not (modeCut and hghtCut), 'Only modeCut or hghtCut can be provided, not both.'
        # TODO checks on hghtCut
        if not (len(self.rsltBck[0]['rv'])==len(self.rsltFwd['rv']) or modeCut or hghtCut): # we need to known how to align the modes...
            modeCut = 0.5 # default, we split fine and coarse at 0.5 μm        
        varsSpctrl = ['aod', 'aodMode', 'n', 'k', 'ssa', 'ssaMode', 'g', 'LidarRatio']
        varsMorph = ['rv', 'sigma', 'sph', 'rEffCalc', 'height']
        varsAodAvg = ['n', 'k'] # modal variables for which we will append a aod weighted average RMSE and BIAS value at the FIRST element (expected to be spectral quantity)
        varsSpctrl = [z for z in varsSpctrl if z in self.rsltFwd] # check that the variable is used in current configuration
        varsMorph = [z for z in varsMorph if z in self.rsltFwd]
        varsAodAvg = [z for z in varsAodAvg if z in self.rsltFwd]
#        TODO: this whole method needs to be rewritten to allow rsltFwd to be a vector
#           it might make sense to break the following code into two methods for len(rsltFwd) > 1 and len(rsltFwd) == 1 (current code basicly)
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
                if modeCut or hghtCut: # use modeCut(hghtCut) to calculated vol weighted contribution to fine(PBL) and coarse(Free Trop.)
                    rtrvdBimode = np.full([rtrvd.shape[0],2], np.nan)
                    if modeCut:
                        for i,rs in enumerate(self.rsltBck): 
                            rtrvdBimode[i]= self.volWghtedAvg(rtrvd[i], rs['rv'], rs['sigma'], rs['vol'], modeCut)
                        true = self.volWghtedAvg(true, self.rsltFwd['rv'], self.rsltFwd['sigma'], self.rsltFwd['vol'], modeCut)
                    else:
                        for i,rs in enumerate(self.rsltBck): 
                            rtrvdBimode[i] = self.hghtWghtedAvg(rtrvd[i], rs['range'], rs['βext'], rs['aodMode'], hghtCut)
                        true = self.hghtWghtedAvg(true, self.rsltFwd['range'], self.rsltFwd['βext'], self.rsltFwd['aodMode'], hghtCut)
                    rtrvd = rtrvdBimode
                else: # a one-to-one pairing of foward and back nodes, requires NmodeFwd==NmodeBack
                    assert len(self.rsltBck[0]['rv'])==len(self.rsltFwd['rv']), 'If modeCut==None, foward and inverted data must have the same number of modes.'
                if av in varsAodAvg: # we also want to calculate the total, mode AOD weighted value of the variable
                    avgVal = [np.sum(rs[av][:,wvlnthInd]*rs['aodMode'][:,wvlnthInd])/np.sum(rs['aodMode'][:,wvlnthInd]) for rs in self.rsltBck]
                    rtrvd = np.vstack([avgVal, rtrvd.T]).T
                    avgVal = np.sum(self.rsltFwd[av][:,wvlnthInd]*self.rsltFwd['aodMode'][:,wvlnthInd])/np.sum(self.rsltFwd['aodMode'][:,wvlnthInd])
                    true = np.r_[avgVal, true]
            if av in ['n','k','sph'] and true.ndim==1 and rtrvd.ndim==1 and true.shape[0]!=rtrvd.shape[0]: # HACK (kinda): if we only retrieve one mode but simulate more we won't report anything
                true = np.mean(true)
#            stdDevCut = 2
#            if rtrvd.ndim==1:
#                rtrvdCln = rtrvd[abs(rtrvd - np.mean(rtrvd)) < stdDevCut * np.std(rtrvd)]
#                rmsErr[av] = np.sqrt(np.mean((true-rtrvdCln)**2, axis=0))
#            else:
#                rmsErr[av] = np.empty(rtrvd.shape[1])
#                for m in range(rtrvd.shape[1]):
#                    rtrvdCln = rtrvd[:,m][abs(rtrvd[:,m] - np.mean(rtrvd[:,m])) < stdDevCut * np.std(rtrvd[:,m])]                        
#                    rmsErr[av][m] = np.sqrt(np.mean((true[m]-rtrvdCln)**2, axis=0))
            rmsErr[av] = np.sqrt(np.median((true-rtrvd)**2, axis=0))
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

    def hghtWghtedAvg(self, val, vertRange, βext, aodMode, hghtCut): 
        wghtsPBL = βext.T[vertRange < hghtCut].sum(axis=0)*aodMode
        wghtsFT = βext.T[vertRange > hghtCut].sum(axis=0)*aodMode
        if np.any(vertRange==hghtCut): warnings.warn('hghtCut fell exactly on one of the vertical bins, this bin was excluded from the weighting.')
        valPBL = np.sum(val*wghtsPBL)/np.sum(wghtsPBL)
        valFT = np.sum(val*wghtsFT)/np.sum(wghtsFT)
        return [valPBL, valFT]

    def ReffMode(self, rs, modeCut):
        if not modeCut: 
            dndr = rs['rv']/np.exp(3*rs['sigma']**2)
            return dndr*np.exp(5/2*rs['sigma']**2) # eq. 60 from Grainger's "Useful Formulae for Aerosol Size Distributions"
        Vfc = self.volWghtedAvg(None, rs['rv'], rs['sigma'], rs['vol'], modeCut)
        Amode = rs['vol']/rs['rv']*np.exp(rs['sigma']**2/2) # convert N to rv and then ratio 1st and 4th rows of Table 1 of Grainger's "Useful Formulae for Aerosol Size Distributions"
        Afc = self.volWghtedAvg(None, rs['rv'], rs['sigma'], Amode, modeCut)
        return Vfc/Afc # NOTE: ostensibly this should be Vfc/Afc/3 but we ommited the factor of 3 from Amode
    
    
        
        
        
