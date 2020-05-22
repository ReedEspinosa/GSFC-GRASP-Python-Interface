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

    def runSim(self, fwdData, bckYAMLpath, Nsims=1, maxCPU=4, binPathGRASP=None, savePath=None, lightSave=False, intrnlFileGRASP=None, releaseYAML=True, rndIntialGuess=False, dryRun=False):
        """ <> runs the simulation for given set of simulated and inversion conditions <>
        fwdData -> yml file path for GRASP fwd model OR "results style" list of dicts
        bckYAMLpath -> yml file path for GRASP inversion
        Nsims -> number of noise pertbations applied to fwd model, must be 1 if fwdData is a list of results dicts
        maxCPU -> the retrieval load will be spread accross maxCPU processes
        binPathGRASP -> path to GRASP binary, if None default from graspRun is used
        savePath -> path to save pickle w/ simulated retrieval results, lightSave -> remove PM data
        intrnlFileGRASP -> alternative path to GRASP kernels, overwrites value in YAML files
        releaseYAML=True -> auto adjust back yaml Nλ to match insturment
        rndIntialGuess=True -> overwrite initial guesses in bckYAMLpath w/ uniformly distributed random values between min & max 
        dryRun -> run foward model and then return noise added graspDB object, without performing the retrievals """
        assert not self.nowPix is None, 'A dummy pixel (nowPix) and error function (addError) are needed in order to run the simulation.' 
        # ADAPT fwdData/RUN THE FOWARD MODEL
        if type(fwdData) == str and fwdData[-3:] == 'yml':
            gObjFwd = rg.graspRun(fwdData)
            gObjFwd.addPix(self.nowPix)
            gObjFwd.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=intrnlFileGRASP)
            try:
                self.rsltFwd = np.array([gObjFwd.readOutput()[0]])
            except IndexError as e:
                raise Exception('Forward calucation output could not be read, halting the simulation.') from e
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
        if not dryRun:
            self.rsltBck = gDB.processData(maxCPU, binPathGRASP, krnlPathGRASP=intrnlFileGRASP, rndGuess=rndIntialGuess)
            assert len(self.rsltBck)>0, 'Inversion output could not be read, halting the simulation (no data was saved).'
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
        else:
            if savePath: warnings.warn('This was a dry run. No retrievals were performed and no results were saved.')
            for gObj in gDB.grObjs: gObj.writeSDATA() 
        return gObjFwd, gDB.grObjs
            
    def loadSim(self, picklePath):
        with open(picklePath, 'rb') as f:
            self.rsltBck = np.array(pickle.load(f))
            try:
                self.rsltFwd = np.array(pickle.load(f))
            except EOFError: # this was an older file (created before Jan 2020)
                self.rsltFwd = [self.rsltBck[-1]] # resltFwd as a array of len==0 (not totaly backward compatible, it used to be straight dict)
                self.rsltBck = self.rsltBck[:-1]
 
    def conerganceFilter(self, χthresh=None, σ=None, verbose=False): # TODO: LIDAR bins with ~0 concentration are dominating this metric...
        """ Only removes data from resltBck if χthresh is provided, χthresh=1.5 seems to work well """
        if σ is None:
            σ={'I'  :0.03, # relative
              'QoI' :0.005, # absolute 
              'UoI' :0.005, # absolute 
              'Q'   :0.005, # absolute in terms of Q/I
              'U'   :0.005, # absolute in terms of U/I
              'LS'  :0.05, # relative
              'VBS' :0.05, # relative
              'VExt':17e-6, # absolute
              }
        for i,rb in enumerate(self.rsltBck):
            rf = self.rsltFwd[0] if len(self.rsltFwd)==1 else self.rsltFwd[i]
            χΤοtal = np.array([])
            for measType in ['VExt', 'VBS', 'LS', 'I', 'QoI', 'UoI', 'Q', 'U']:
                fitKey = 'fit_'+measType
                if fitKey in rb:
                    DBck = rb[fitKey][~np.isnan(rb[fitKey])] 
                    if fitKey in rf: 
                         DFwd = rf[fitKey][~np.isnan(rf[fitKey])]
                    elif measType in ['QoI', 'UoI'] and fitKey[:-2] in rf: # rf has X while rb has XoI
                         DFwd = rf[fitKey[:-2]][~np.isnan(rf[fitKey[:-2]])]/rf['fit_I'][~np.isnan(rf[fitKey[:-2]])]
                    if measType in ['I', 'LS', 'VBS']: # relative errors
                        with np.errstate(divide='ignore'): # possible for DBck+DFwd=0, inf's will be removed below
                            χLocal = ((2*(DBck-DFwd)/(DBck+DFwd))/σ[measType])**2
                        χLocal[χLocal>100] = 100 # cap at 10σ (small values may produce huge relative errors)
                    else: # absolute errors
                        if measType in ['Q', 'U']: # we need to normalize by I, all Q[U] errors given in terms of q[u]
                            DBck = DBck/rb['fit_I'][~np.isnan(rb[fitKey])]
                            DFwd = DFwd/rf['fit_I'][~np.isnan(rf[fitKey])]
                        χLocal = ((DBck-DFwd)/σ[measType])**2
                    χΤοtal = np.r_[χΤοtal, χLocal]   
            rb['χ2'] = np.sqrt(np.mean(χΤοtal))
        if χthresh and len(self.rsltBck) > 2: # we will always keep at least 2 entries
            validInd = np.array([rb['χ2']<=χthresh for rb in self.rsltBck])
            if verbose: print('%d/%d met χthresh' % (validInd.sum(), len(self.rsltBck)))
            if validInd.sum() < 2:
                validInd = np.argsort([rb['χ2'] for rb in self.rsltBck])[0:2] # note validInd went from bool to array of ints
                if verbose:
                    print('Preserving the two rsltBck elements with lowest χ scores, even though they did not meet χthresh.')
            self.rsltBck = self.rsltBck[validInd]
        elif χthresh and np.sum([rb['χ2']<=χthresh for rb in self.rsltBck])<2 and verbose:
            print('rsltBck only has two or fewer elements, no χthresh screening will be perofmed.')
                    
    def analyzeSim(self, wvlnthInd=0, modeCut=None, hghtCut=None, fineModesFwd=None, fineModesBck=None): 
        """ Returns the RMSE and bias (defined below) from the simulation results
                wvlngthInd - the index of the wavelength to calculate stats for
                modeCut - fine/coarse seperation radius in um, currenltly only applied to rEff (None -> do not calculate error's modal dependence)
                hghtCut - PBL/FT seperation in meters (None -> do not calculate error's layer dependence)
                fineModesFwd - [array-like] the indices of the fine modes in the foward calculation, set to None to use OSSE ..._Fine variables instead 
                fineModesBck -  [array-like] the indices of the fine modes in the retrieval
                NOTE: this method generally assumes configuration (e.g. # of modes) is the same across all pixels
                TODO: we can do a lot better than we do here in terms of modeCut and hghtCut...
                    - rEff needs to include seperate errors for PBL and column/FT, if it comes out now it is probably just AOD weighted (which isn't exact)
                    - modeCut should be applied to more variables (e.g. spherical fraction)
                    - we can also do pbl/FT g and LidarRatio exactly (instead of just AOD weighted)
                        - formulas are less trivial but see/merge with rtrvdDataSetPixels() in readOSSEnetCDF
                        - note that these are only exact for a single species (i.e. one CRI, PSD, etc.), ultimatly we need to save modal g and lidarRatio
                """
        # check on input and available variables
        assert (not self.rsltBck is None) and (not self.rsltFwd is None), 'You must call loadSim() or runSim() before you can calculate statistics!'
        if type(self.rsltFwd) is dict: self.rsltFwd = [self.rsltFwd]
        assert type(self.rsltFwd) is list or type(self.rsltFwd) is np.ndarray, 'rsltFwd must be a list! Note that it was stored as a dict in older versions of the code.'
        fwdKys = self.rsltFwd[0].keys()
        bckKys = self.rsltBck[0].keys()
        lgnrmPSD = ('rv' in fwdKys or 'aod_Fine' in fwdKys) and 'rv' in self.rsltBck[0]
        assert modeCut is None or lgnrmPSD, 'Fine/Coarse errors can only be caluclated from GRASPs lognormal PSD representation! For this data, you must set modeCut=None' 
        hghtInfo = ('βext' in fwdKys or 'aod_PBL' in fwdKys) and 'βext' in self.rsltBck[0]
        assert hghtCut is None or hghtInfo, 'PBL/FT errors can only currently be calculated from LIDAR retrievals! For this retrieval dataset, you must set heghtCut=None' 
        assert 'aodMode' not in fwdKys or fineModesFwd is None or self.rsltFwd[0]['aodMode'].shape[0] > max(fineModesFwd), 'fineModesFwd contains indices that are too high given the number of modes in rsltFwd[aodMode]'
        assert 'aodMode' not in bckKys or fineModesBck is None or self.rsltBck[0]['aodMode'].shape[0] > max(fineModesBck), 'fineModesBck contains indices that are too high given the number of modes in rsltBck[aodMode]'        
        # define functions for calculating RMS and bias
        rmsFun = lambda t,r: np.sqrt(np.median((t-r)**2, axis=0)) # formula for RMS output (true->t, retrieved->r)
        biasFun = lambda t,r: r-t if r.ndim > 1 else np.atleast_2d(r-t).T # formula for bias output
        # variables we expect to see
        varsSpctrl = ['aod', 'aodMode', 'n', 'k', 'ssa', 'ssaMode', 'g', 'LidarRatio']
        varsMorph = ['rv', 'sigma', 'sph', 'rEffCalc', 'height']
        varsAodAvg = ['n', 'k'] # modal variables for which we will append aod weighted average RMSE and BIAS value at the FIRST element (expected to be spectral quantity)
        modalVars = ['rv', 'sigma', 'sph', 'aodMode', 'ssaMode'] + varsAodAvg # variables for which we find fine/coarse or FT/PBL errors seperately  
        varsSpctrl = [z for z in varsSpctrl if z in fwdKys and z in bckKys] # check that the variable is used in current configuration
        varsMorph = [z for z in varsMorph if z in fwdKys and z in bckKys]
        # loop through varsSpctrl and varsMorph calcualted RMS and bias
        rmsErr = dict()
        bias = dict()
        for av in varsSpctrl+varsMorph:
            rtrvd = np.array([rb[av] for rb in self.rsltBck])
            true = np.array([rf[av] for rf in self.rsltFwd])
            if av in varsSpctrl:
                rtrvd = rtrvd[...,wvlnthInd]
                true = true[...,wvlnthInd]
            if rtrvd.ndim==1: rtrvd = np.expand_dims(rtrvd,1) # we want [ipix, imode], we add a singleton dimension if only one mode/nonmodal 
            if true.ndim==1: true = np.expand_dims(true,1) # we want [ipix, imode], we add a singleton dimension if only one mode/nonmodal             
            if hghtCut and av in modalVars and (av+'_PBL' in fwdKys or 'aodMode' in fwdKys): # calculate vertical dependent RMS/BIAS [PBL, FT*]
                if av+'_PBL' in fwdKys: # TODO: we have foward model PBL height... we should use it
                    trueBilayer = self.getStateVals(av+'_PBL', self.rsltFwd, varsSpctrl, wvlnthInd)
                    rtrvdBilayer = self.hghtWghtedAvg(rtrvd, self.rsltBck, wvlnthInd, hghtCut, pblOnly=True)
                else: # 'aodMode' in self.rsltFwd[0]
                    trueBilayer = self.hghtWghtedAvg(true, self.rsltFwd, wvlnthInd, hghtCut)
                    rtrvdBilayer = self.hghtWghtedAvg(rtrvd, self.rsltBck, wvlnthInd, hghtCut)
                rmsErr[av+'_PBLFT'] = rmsFun(trueBilayer, rtrvdBilayer) # PBL is 1st ind, FT (not total column!) is 2nd
                bias[av+'_PBLFT'] =  biasFun(trueBilayer, rtrvdBilayer)
            if not fineModesBck is None and av in modalVars: # calculate fine mode dependent RMS/BIAS
                fineCalculated = False
                if av+'_Fine' in fwdKys and fineModesFwd is None and 'aodMode' in bckKys: # we have OSSE outputs, currently user provided fineModesFwd trumps this though
                    trueFine = self.getStateVals(av+'_Fine', self.rsltFwd, varsSpctrl, wvlnthInd)
                    rtrvdFine = self.getStateVals(av, self.rsltBck, varsSpctrl, wvlnthInd, fineModesBck)
                    fineCalculated = True
                elif not fineModesFwd is None and 'aodMode' in fwdKys and 'aodMode' in bckKys: # user provided fwd and bck fine mode indices
                    trueFine = self.getStateVals(av, self.rsltFwd, varsSpctrl, wvlnthInd, fineModesFwd)
                    rtrvdFine = self.getStateVals(av, self.rsltBck, varsSpctrl, wvlnthInd, fineModesBck)
                    fineCalculated = True                    
                elif not fineModesFwd is None: # something went wrong...
                    assert False, 'If fineModeFwd and fineModeBck are provided aodMode must be present in rsltsBck and rsltsFwd!'
                if fineCalculated:
                    rmsErr[av+'_fine'] = rmsFun(trueFine, rtrvdFine) # PBL is 1st ind, FT (not total column!) is 2nd
                    bias[av+'_fine'] =  biasFun(trueFine, rtrvdFine)
            if av in varsAodAvg: # calculate the total, mode AOD weighted value of the variable (likely just CRI) -> [total, mode1, mode2,...]
                rtrvd = np.hstack([self.τWghtedAvg(rtrvd, self.rsltBck, wvlnthInd), rtrvd])
                true = np.hstack([self.τWghtedAvg(true, self.rsltFwd, wvlnthInd), true])                    
            if true.shape[1] == rtrvd.shape[1]: # truth and retrieved modes can be paired one-to-one    
                rmsErr[av] = rmsFun(true, rtrvd) # BUG: fineModesFwd and fineModesBck and not taken into accoutn here, really we just shouldn't return n or k with more than one mode (we have n(k)_fine now)
                bias[av] = biasFun(true, rtrvd)
            elif av in varsAodAvg and 'aodMode' in fwdKys and 'aodMode' in bckKys: # we at least know the first, total elements of CRI correspond to each other
                rmsErr[av] = rmsFun(true[:,0], rtrvd[:,0])
                bias[av] = biasFun(true[:,0], rtrvd[:,0])
            if modeCut: # calculate rEff, could be abstracted into above code but tricky b/c volume weighted mean will not give exactly correct results (code below is exact)
                rtrvd = np.array([self.ReffMode(rb, modeCut) for rb in self.rsltBck])
                true = np.array([self.ReffMode(rf, modeCut) for rf in self.rsltFwd])
                rmsErr['rEff_sub%dnm' % 1000*modeCut] = rmsFun(true, rtrvd)
                bias['rEff_sub%dnm' % 1000*modeCut] = biasFun(true, rtrvd)
                # TODO: add sph fraction here? It would use volWghtedAvg, I think in its exact current form
        return rmsErr, bias
    
    def getStateVals(self, av, rslts, varsSpctrl, wvlnthInd, modeInd=None):
        assert av in rslts[0], 'Variable %s was not found in the rslts dictionary!' % av
        stateVals = np.array([rf[av] for rf in rslts])
        if av in varsSpctrl:
            stateVals = stateVals[...,wvlnthInd]
        if not modeInd is None and 'aod' in av: # we will sum multiple fine modes to get total fine mode AOD
            stateVals = np.expand_dims(stateVals[:, modeInd].sum(axis=1),1)
        elif not modeInd is None: # we will perform AOD weighted averaging from all fine modes of an intensive property
            stateVals = self.τWghtedAvg(stateVals[:, modeInd], rslts, wvlnthInd, modeInd)
        return stateVals

    def τWghtedAvg(self, val, rslts, wvlnthInd, modeInd=slice(None)):
        avgVal = np.full(len(rslts), np.nan)
        if (val.shape[1]==1 and slice(None)==modeInd) or not 'aodMode' in rslts[0]: # there was only one mode to start OR can't calculate b/c we don't have aodMode
            return np.zeros([val.shape[0], 0]) # return an empty array
        for i,rslt in enumerate(rslts):
            ttlSum = np.sum(val[i]*rslt['aodMode'][modeInd,wvlnthInd])
            normDenom = np.sum(rslt['aodMode'][modeInd,wvlnthInd])
            avgVal[i] = ttlSum/normDenom
        return np.expand_dims(avgVal, 1)
    
    def hghtWghtedAvg(self, val, rslts, wvlnthInd, hghtCut, pblOnly=False): 
        Bilayer = np.full([len(rslts), 2-pblOnly], np.nan)
        if np.any(rslts[0]['range']==hghtCut): warnings.warn('hghtCut fell exactly on one of the vertical bins, this bin will be excluded from the weighting.')
        for i,rslt in enumerate(rslts): 
            wghtsPBL = []
            for β,h,τ in zip(rslt['βext'], rslt['range'], rslt['aodMode'][:,wvlnthInd]): # loop over modes
                wghtsPBL.append(β[h < hghtCut].sum()/β.sum()*τ)
            valPBL = np.sum(val*wghtsPBL)/np.sum(wghtsPBL)
            if not pblOnly:
                wghtsFT = []
                for β,h,τ in zip(rslt['βext'], rslt['range'], rslt['aodMode'][:,wvlnthInd]):
                    wghtsFT.append(β[h > hghtCut].sum()/β.sum()*τ)
                    valFT = np.sum(val*wghtsFT)/np.sum(wghtsFT)
                Bilayer[i] = [valPBL, valFT]
            else:
                Bilayer[i] = [valPBL]
        return Bilayer

    def ReffMode(self, rs, modeCut):
        if not modeCut: 
            rv_dndr = rs['rv']/np.exp(3*rs['sigma']**2)
            return rv_dndr*np.exp(5/2*rs['sigma']**2) # eq. 60 from Grainger's "Useful Formulae for Aerosol Size Distributions"
        Vfc = self.volWghtedAvg(None, [rs], modeCut)
        Amode = rs['vol']/rs['rv']*np.exp(rs['sigma']**2/2) # convert N to rv and then ratio 1st and 4th rows of Table 1 of Grainger's "Useful Formulae for Aerosol Size Distributions"
        Afc = self.volWghtedAvg(None, [rs], modeCut, Amode)
        return Vfc/Afc # NOTE: ostensibly this should be Vfc/Afc/3 but we ommited the factor of 3 from Amode as well
    
    def volWghtedAvg(self, val, rslts, modeCut, vol=None, fineOnly=False):
        N = 1e3 # number of radii bins
        lower = [modeCut/50] if fineOnly else [modeCut/50, modeCut]
        upper = [modeCut] if fineOnly else [modeCut, modeCut*50]
        Bimode = np.full([len(rslts), 2-fineOnly], np.nan)
        for i,rslt in enumerate(rslts): # loop over each pixel/time
            crsWght = [] # this will be [upr/lwr(N=2), mode]
            for upr, lwr, in zip(upper,lower):
                r = np.logspace(np.log10(lwr),np.log10(upr),N)
                crsWght.append([np.trapz(ms.logNormal(mu, σ, r)[0],r) for mu,σ in zip (rslt['rv'],rslt['sigma'])]) # integrated r 0->inf this will sum to unity
            if not np.isclose(crsWght.sum(axis=0), 1, rtol=0.001).all():
                warnings.warn('The sum of the crsWght values across all modes was greater than 0.1%% from unity')
            if val is None: # we just want volume (or area if vol arugment contains area) concentration of each mode
                if vol is None:
                    crsWght = np.array(crsWght)*rslt['vol'] 
                else:
                    crsWght = np.array(crsWght)*vol
                Bimode[i] = np.sum(crsWght, axis=1)
            else: 
                Bimode[i] = np.sum(crsWght*val[i], axis=1)
        return Bimode

    
        
        
        
