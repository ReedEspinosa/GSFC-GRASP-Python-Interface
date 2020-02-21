#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
import pickle
import os
import warnings
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
                self.rsltFwd = [self.rsltBck[-1]] # resltFwd as a array of len==0 (not totaly backward compatible, it used to be straight dict)
                self.rsltBck = self.rsltBck[:-1]
 
    def analyzeSim(self, wvlnthInd=0, modeCut=None, hghtCut=None): 
        """ Returns the RMSE and bias (defined below) from the simulation results
                wvlngthInd - the index of the wavelength to calculate stats for
                modeCut - fine/coarse seperation radius in um (None -> do not calculate error's modal dependence)
                hghtCut - PBL/FT seperation in meters (None -> do not calculate error's layer dependence)
                NOTE: this method generally assumes configuration (e.g. # of modes) is the same across all pixels
                """
        assert type(self.rsltFwd) is list or type(self.rsltFwd) is np.ndarray, 'rsltFwd must be a list! Note that it was stored as a dict in older versions of the code.'
        lgnrmPSD = ('rv' in self.rsltFwd[0] and 'rv' in self.rsltBck[0])
        assert modeCut is None or lgnrmPSD, 'Fine/Coarse errors can only be caluclated from GRASPs lognormal PSD representation! For this data, you must set modeCut=None' 
        assert hghtCut is None or ('βext' in self.rsltFwd[0] and 'βext' in self.rsltBck[0]), 'PBL/FT errors can only be calculated LIDAR retrievals! For this data, you must set heghtCut=None' 
        rmsFun = lambda t,r: np.sqrt(np.median((t-r)**2, axis=0)) # formula for RMS output (true->t, retrieved->r)
        biasFun = lambda t,r: r-t if r.ndim > 1 else np.atleast_2d(r-t).T # formula for bias output
        varsSpctrl = ['aod', 'aodMode', 'n', 'k', 'ssa', 'ssaMode', 'g', 'LidarRatio']
        varsMorph = ['rv', 'sigma', 'sph', 'rEffCalc', 'height']
        varsAodAvg = ['n', 'k'] # modal variables for which we will append aod weighted average RMSE and BIAS value at the FIRST element (expected to be spectral quantity)
        varsSpctrl = [z for z in varsSpctrl if z in self.rsltFwd[0]] # check that the variable is used in current configuration
        varsMorph = [z for z in varsMorph if z in self.rsltFwd[0]]
        varsAodAvg = [z for z in varsAodAvg if z in self.rsltFwd[0]]
        rmsErr = dict()
        bias = dict()
        assert (not self.rsltBck is None) and (not self.rsltFwd is None), 'You must call loadSim() or runSim() before you can calculate statistics!'
        for av in varsSpctrl+varsMorph+['rEffMode'] if lgnrmPSD else varsSpctrl+varsMorph:
            if av=='rEffMode':
                rtrvd = np.array([self.ReffMode(rb, modeCut) for rb in self.rsltBck])
                true = np.array([self.ReffMode(rf, modeCut) for rf in self.rsltFwd])
            else:
                rtrvd = np.array([rb[av] for rb in self.rsltBck])
                true = np.array([rf[av] for rf in self.rsltFwd])
                if av in varsSpctrl:
                    rtrvd = rtrvd[...,wvlnthInd]
                    true = true[...,wvlnthInd]
            if rtrvd.ndim>1 and not av=='rEffMode': # we will seperate fine/coarse or FT/PBL modes here (rtrvd.ndim==1 for nonmodal vars, e.g. AOD) 
                if modeCut: # use modeCut to calculate vol weighted contribution to fine and coarse
                    rtrvdBimode = self.volWghtedAvg(rtrvd, self.rsltBck, modeCut)
                    trueBimode = self.volWghtedAvg(true, self.rsltFwd, modeCut)
                if hghtCut: # use hghtCut to calculate vol weighted contribution to PBL and Free Trop.
                    rtrvdBilayer = self.hghtWghtedAvg(rtrvd, self.rsltBck, wvlnthInd, hghtCut)
                    trueBilayer = self.hghtWghtedAvg(true, self.rsltFwd, wvlnthInd, hghtCut)
                if av in varsAodAvg: # calculate the total, mode AOD weighted value of the variable (likely just CRI) -> [total, mode1, mode2,...]
                    rtrvd = np.vstack([self.τWghtedAvg(av, self.rsltBck, wvlnthInd), rtrvd.T]).T
                    true = np.vstack([self.τWghtedAvg(av, self.rsltFwd, wvlnthInd), true.T]).T                    
            # cacluate RMS and bias to be returned
            if rtrvd.ndim>1 and not av=='rEffMode' and modeCut:
                rmsErr[av+'_finecoarse'] = rmsFun(trueBimode, rtrvdBimode) # fine is 1st ind, coarse is 2nd
                bias[av+'_finecoarse'] = biasFun(trueBimode, rtrvdBimode)
            if rtrvd.ndim>1 and not av=='rEffMode' and hghtCut:
                rmsErr[av+'_PBLFT'] = rmsFun(trueBilayer, rtrvdBilayer) # PBL is 1st ind, FT (not total column!) is 2nd
                bias[av+'_PBLFT'] =  biasFun(trueBilayer, rtrvdBilayer)      
            if true.shape[1]==rtrvd.shape[1]: # truth and retrieved modes can be paired one-to-one    
                rmsErr[av] = rmsFun(true, rtrvd)
                bias[av] = biasFun(true, rtrvd)
        return rmsErr, bias
    
    def volWghtedAvg(self, val, rslts, modeCut, vol=None):
        N = 1e3
        lower = [modeCut/50, modeCut]
        upper = [modeCut, modeCut*50]
        Bimode = np.full([len(rslts),2], np.nan)
        for i,rslt in enumerate(rslts): 
            crsWght = []
            for upr, lwr, in zip(upper,lower):
                r = np.logspace(np.log10(lwr),np.log10(upr),N)
                crsWght.append([np.trapz(ms.logNormal(mu, σ, r)[0],r) for mu,σ in zip (rslt['rv'],rslt['sigma'])])
            crsWght = np.array(crsWght)*rslt['vol'] if vol is None else np.array(crsWght)*vol
            if val is None: # we just want volume (or area if vol arugment contains area) concentration of each mode
                Bimode[i] = np.sum(crsWght,axis=1)
            else: 
                Bimode[i] = np.sum(crsWght*val[i],axis=1)/np.sum(crsWght,axis=1)
        return Bimode

    def hghtWghtedAvg(self, val, rslts, wvlnthInd, hghtCut): 
        Bilayer = np.full([len(rslts),2], np.nan)
        for i,rslt in enumerate(rslts): 
            wghtsPBL = []
            wghtsFT = []
            for β,h,τ in zip(rslt['βext'], rslt['range'], rslt['aodMode'][:,wvlnthInd]):
                wghtsPBL.append(β[h < hghtCut].sum()/β.sum()*τ)
                wghtsFT.append(β[h > hghtCut].sum()/β.sum()*τ)
            if np.any(rslt['range']==hghtCut): warnings.warn('hghtCut fell exactly on one of the vertical bins, this bin was excluded from the weighting.')
            valPBL = np.sum(val*wghtsPBL)/np.sum(wghtsPBL)
            valFT = np.sum(val*wghtsFT)/np.sum(wghtsFT)
            Bilayer[i] = [valPBL, valFT]
        return Bilayer

    def τWghtedAvg(self, av, rslts, wvlnthInd):
        avgVal = np.full(len(rslts), np.nan)
        for i,rslt in enumerate(rslts):
            ttlSum = np.sum(rslt[av][:,wvlnthInd]*rslt['aodMode'][:,wvlnthInd])
            normDenom = np.sum(rslt['aodMode'][:,wvlnthInd])
            avgVal[i] = ttlSum/normDenom
        return avgVal

    def ReffMode(self, rs, modeCut):
        if not modeCut: 
            dndr = rs['rv']/np.exp(3*rs['sigma']**2)
            return dndr*np.exp(5/2*rs['sigma']**2) # eq. 60 from Grainger's "Useful Formulae for Aerosol Size Distributions"
        Vfc = self.volWghtedAvg(None, [rs], modeCut)
        Amode = rs['vol']/rs['rv']*np.exp(rs['sigma']**2/2) # convert N to rv and then ratio 1st and 4th rows of Table 1 of Grainger's "Useful Formulae for Aerosol Size Distributions"
        Afc = self.volWghtedAvg(None, [rs], modeCut, Amode)
        return Vfc/Afc # NOTE: ostensibly this should be Vfc/Afc/3 but we ommited the factor of 3 from Amode
    
    
        
        
        
