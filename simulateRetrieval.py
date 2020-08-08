#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pickle
import shutil
import os
import warnings
import datetime as dt
import numpy as np
from scipy.stats import norm
import runGRASP as rg
import miscFunctions as ms


class simulation(object):
    def __init__(self, nowPix=None, picklePath=None):
        assert not (nowPix and picklePath), 'Either nowPix or picklePath should be provided, but not both.'
        if picklePath: self.loadSim(picklePath)
        if nowPix is None:
            self.nowPix = None
            return
        assert np.all([np.all(np.diff(mv['meas_type'])>0) for mv in nowPix.measVals]), 'nowPix.measVals[l][\'meas_type\'] must be in ascending order at each l'
        self.nowPix = copy.deepcopy(nowPix) # we will change this, bad form not to make our own copy
        self.rsltBck = None
        self.rsltFwd = None

    def runSim(self, fwdData, bckYAMLpath, Nsims=1, maxCPU=4, maxT=None, binPathGRASP=None, savePath=None,
               lightSave=False, intrnlFileGRASP=None, releaseYAML=True, rndIntialGuess=False,
               dryRun=False, workingFileSave=False, fixRndmSeed=False, radianceNoiseFun=None, verbose=False):
        """
        <> runs the simulation for given set of simulated and inversion conditions <>
        fwdData -> yml file path for GRASP fwd model OR "results style" list of dicts
        bckYAMLpath -> yml file path for GRASP inversion
        Nsims -> number of noise pertbations applied to fwd model, must be 1 if fwdData is a list of results dicts
        maxCPU -> the retrieval load will be spread accross maxCPU processes
        binPathGRASP -> path to GRASP binary, if None default from graspRun is used
        savePath -> path to save pickle w/ simulated retrieval results, lightSave -> remove PM data to save space
        intrnlFileGRASP -> alternative path to GRASP kernels, overwrites value in YAML files
        releaseYAML=True -> auto adjust back yaml Nλ and number of vertical bins to match the forward simulated data
        rndIntialGuess=True -> overwrite initial guesses in bckYAMLpath w/ uniformly distributed random values between min & max
        dryRun -> run foward model and then return noise added graspDB object, without performing the retrievals
        workingFileSave -> create ZIP with the GRASP SDATA, YAML and Output files used in the run, saved to savePath + .zip
        fixRndmSeed -> Use same random seed for the measurement noise added (each pixel will have identical noise values)
                            Only works if nowPix.measVals[n]['errorModel'] uses the `random` module to generate noise for all n
        radianceNoiseFun -> a function with 1st arg λ (μm), 2nd arg rslt dict & 3rd arg verbose bool to map rslt fit_I/Q/U to retrieval (back) SDATA
                                See addError() at bottom of architectureMap in ACCP folding of MADCAP scripts for an example
                                This option will override an error model in nowPix; set to None add no noise to OSSE polarimeter
        """
        if fixRndmSeed and not rndIntialGuess:
            warnings.warn('Identical noise values and initial guess used in each pixel, repeating EXACT same retrieval %d times!' % Nsims)
        # ADAPT fwdData/RUN THE FOWARD MODEL
        if type(fwdData) == str and fwdData[-3:] == 'yml': # we are using GRASP's fwd model
            assert self.nowPix is not None, 'A dummy pixel (nowPix) with an error function (addError) is required to run the simulation.'
            if verbose: print('Calculating forward model "truth"...')
            gObjFwd = rg.graspRun(fwdData)
            gObjFwd.addPix(self.nowPix)
            gObjFwd.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=intrnlFileGRASP)
            try:
                self.rsltFwd = np.array([gObjFwd.readOutput()[0]])
            except IndexError as e:
                raise Exception('Forward calucation output could not be read, halting the simulation.') from e
            loopInd = np.zeros(Nsims, int)
        elif type(fwdData) == list: # likely OSSE from netCDF
            self.rsltFwd = fwdData
            gObjFwd = rg.graspRun()
            gObjFwd.invRslt = fwdData
            assert Nsims <= 1, 'Running multiple noise perturbations on more than one foward simulation is not supported.'
            loopInd = range(len(self.rsltFwd))
            if self.nowPix is None: self.nowPix = rg.pixel() # nowPix is optional argument to _init_ in OSSE case
        else:
            assert False, 'Unrecognized data type, fwdModelYAMLpath should be path to YAML file or a DICT!'
        if verbose: print('Forward model "truth" obtained')
        # ADD NOISE AND PERFORM RETRIEVALS
        if verbose: print('Inverting noised-up measurements...')
        gObjBck = rg.graspRun(bckYAMLpath, releaseYAML=releaseYAML, quietStart=verbose) # quietStart=True -> we won't see path of temp, pre-gDB graspRun
        if fixRndmSeed: strtSeed = np.random.randint(low=0, high=2**32-1)
        localVerbose = verbose
        for tOffset, i in enumerate(loopInd): # loop over each simulated pixel, later split up into maxCPU calls to GRASP
            if fixRndmSeed: np.random.seed(strtSeed) # reset to same seed, adding same noise to every pixel
            self.nowPix.populateFromRslt(self.rsltFwd[i], radianceNoiseFun=radianceNoiseFun, verbose=localVerbose)
            if len(np.unique(loopInd)) != len(loopInd): # we are using the same rsltFwd dictionary more than once
                self.nowPix.dtObj = self.nowPix.dtObj + dt.timedelta(seconds=tOffset) # increment hour otherwise GRASP will whine
            gObjBck.addPix(self.nowPix) # addPix performs a deepcopy on nowPix, won't be impact by next iteration through loopInd
            localVerbose = False # verbose output for just one pixel should be sufficient
        gDB = rg.graspDB(gObjBck, maxCPU=maxCPU, maxT=maxT)
        if not dryRun:
            self.rsltBck = gDB.processData(maxCPU, binPathGRASP, krnlPathGRASP=intrnlFileGRASP, rndGuess=rndIntialGuess)
            assert len(self.rsltBck)>0, 'Inversion output could not be read, halting the simulation (no data was saved).'
            # SAVE RESULTS
            if savePath: self.saveSim(savePath, lightSave, verbose)
        else:
            if savePath: warnings.warn('This was a dry run. No retrievals were performed and no results were saved.')
            for gObj in gDB.grObjs: gObj.writeSDATA()
        if workingFileSave and savePath: # TODO: build zip from original tmp folders without making extra copies to disk, see first answer here: https://stackoverflow.com/questions/458436/adding-folders-to-a-zip-file-using-python
            fullSaveDir = savePath[0:-4]
            if verbose: print('Packing GRASP working files up into %s' %  fullSaveDir + '.zip')
            if os.path.exists(fullSaveDir): shutil.rmtree(fullSaveDir)
            os.mkdir(fullSaveDir)
            if self.dirGRASP is not None: # If yes, then this was an OSSE run (no forward calculation folder)
                shutil.copytree(gObjFwd.dirGRASP, os.path.join(fullSaveDir,'forwardCalculation'))
            for i, gb in enumerate(gDB.grObjs):
                shutil.copytree(gb.dirGRASP, os.path.join(fullSaveDir,'inversion%02d' % i))
            shutil.make_archive(fullSaveDir, 'zip', fullSaveDir)
            shutil.rmtree(fullSaveDir)
        return gObjFwd, gDB.grObjs

    def saveSim(self, savePath, lightSave=False, verbose=False):
        if not os.path.exists(os.path.dirname(savePath)):
            print('savePath (%s) did not exist, creating it...')
            os.makedirs(os.path.dirname(savePath))
        if lightSave:
            for pmStr in ['p11','p12','p22','p33','p34','p44']:
                [rb.pop(pmStr, None) for rb in self.rsltBck]
                if len(self.rsltFwd) > 1: [rf.pop(pmStr, None) for rf in self.rsltFwd]
        if verbose: print('Saving simulation results to %s' %  savePath)
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

    def conerganceFilter(self, χthresh=None, σ=None, forceχ2Calc=False, verbose=False, minSaved=2): # TODO: LIDAR bins with ~0 concentration are dominating this metric...
        """ Only removes data from resltBck if χthresh is provided, χthresh=1.5 seems to work well
        Now we use costVal from GRASP if available (or if forceχ2Calc==True), χthresh≈2.5 is probably better
        NOTE: if forceχ2Calc==True or χthresh~=None this will permanatly alter the values of rsltBck/rsltFwd
        """
        if σ is None:
            σ={'I'  :0.03, # relative
               'QoI' :0.005, # absolute
               'UoI' :0.005, # absolute
               'Q'   :0.005, # absolute in terms of Q/I
               'U'   :0.005, # absolute in terms of U/I
               'LS'  :0.05, # relative
               'VBS' :0.05, # relative
               'VExt':17e-6} # absolute
        if 'costVal' not in self.rsltBck[0] or forceχ2Calc:
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
                rb['costVal'] = np.sqrt(np.mean(χΤοtal))
        if χthresh and len(self.rsltBck) > 2: # we will always keep at least 2 entries
            validInd = np.array([rb['costVal']<=χthresh for rb in self.rsltBck])
            if verbose: print('%d/%d met χthresh' % (validInd.sum(), len(self.rsltBck)))
            if validInd.sum() < minSaved:
                validInd = np.argsort([rb['costVal'] for rb in self.rsltBck])[0:minSaved] # note validInd went from bool to array of ints
                if verbose:
                    print('Preserving the %d rsltBck elements with lowest χ scores, even though they did not meet χthresh.' % minSaved)
            if len(self.rsltFwd)==len(self.rsltBck): self.rsltFwd = self.rsltFwd[validInd]
            self.rsltBck = self.rsltBck[validInd]
        elif χthresh and np.sum([rb['costVal']<=χthresh for rb in self.rsltBck])<minSaved and verbose:
            print('rsltBck only has %d or fewer elements, no χthresh screening will be perofmed.' % minSaved)

    def analyzeSimProfile(self, wvlnthInd=0, FineModes=[0,2]):
        """ We will assume:
        len(rsltFwd)==len(rsltBck) << CHECKED BELOW
        finemodes and pbl modes are same index in fwd and bck data

        We need error/bias estimats for:
        number concentration FT/PBL resolved
        effective radius FT/PBL resolved
            --- reff vertical averaging (reff_h=Rh, # conc.=Nh, vol_conc.=Vh, height_ind=h) ---
            reff = ∫ ΣNh*r^3*dr/∫ ΣNh*r^2*dr = 3/4/π*ΣVh/∫ Σnh*r^2*dr
            Rh = (3/4/π*Vh)/∫ nh*r^2*dr -> reff = 3/4/π*ΣVh/(Σ(3/4/π*Vh)/Rh) = ΣVh/Σ(Vh/Rh) = Σβh/Σ(βh/Rh)
        ssa FT/PBL resolved # ω=Σβh/Σαh & ωh*αh=βh => ω=Σωh*αh/Σαh
        lidar ratio FT/PBL resolved  # S=τ/F11[180] & F11h[180]=τh/Sh => S=Στh/Σ(τh/Sh) <<< not implemented yet
        """
        assert len(self.rsltFwd)==len(self.rsltBck), 'This method only works with sims where fwd and bck pair one-to-one'
        if 'βext' not in self.rsltFwd[0] or not len(self.rsltFwd[0]['aodMode'][:,0])==len(self.rsltBck[0]['aodMode'][:,0]): # return empty dicts if this is polarimeter only run
            return dict(), dict(), dict()
        pxKy = ['biasExt','trueExt','biasExtFine','trueExtFine', 'biasSSA', 'trueSSA']
        pxDct = {k: [] for k in pxKy}
        for rf,rb in zip(self.rsltFwd, self.rsltBck):
            Nmodes = len(rf['aodMode'][:,0])
            assert Nmodes==len(rb['aodMode'][:,0]), 'Fwd and bck should pair one-to-one but they had different lengths!'
            rng = rf['range'][0,:]
            Nrange = len(rng)
            extFwd = np.empty([Nmodes, Nrange])
            extBck = np.empty([Nmodes, Nrange])
            scaFwd = np.empty([Nmodes, Nrange])
            scaBck = np.empty([Nmodes, Nrange])
            for i in range(Nmodes):
                extFwd[i,:] = ms.norm2absExtProf(rf['βext'][i,:], rng, rf['aodMode'][i,wvlnthInd])
                extBck[i,:] = ms.norm2absExtProf(rb['βext'][i,:], rng, rb['aodMode'][i,wvlnthInd])
                scaFwd[i,:] = extFwd[i,:]*rf['ssaMode'][i,wvlnthInd]
                scaBck[i,:] = extBck[i,:]*rb['ssaMode'][i,wvlnthInd]
            pxDct['biasExt'].append(np.sum(extBck - extFwd, axis=0)) # Σa-b = Σa - Σb
            pxDct['trueExt'].append(np.sum(extFwd, axis=0)) # sum over all modes
            pxDct['biasExtFine'].append(np.sum(extBck[FineModes,:] - extFwd[FineModes,:], axis=0)) # Σa-b = Σa - Σb
            pxDct['trueExtFine'].append(np.sum(extFwd[FineModes,:], axis=0)) # sum over all modes
            ssaTr = np.sum(scaFwd, axis=0)/np.sum(extFwd, axis=0)
            ssaBck = np.sum(scaBck, axis=0)/np.sum(extBck, axis=0)
            pxDct['biasSSA'].append(ssaBck - ssaTr) # Σa-b = Σa - Σb
            pxDct['trueSSA'].append(ssaTr)
        bias = {'βext':np.array(pxDct['biasExt']), 'βextFine':np.array(pxDct['biasExtFine']),
                'ssa':np.array(pxDct['biasSSA'])}
        rmse = {'βext':np.sqrt(np.mean(bias['βext']**2, axis=0)),
                'βextFine':np.sqrt(np.mean(bias['βextFine']**2, axis=0)),
                'ssa':np.sqrt(np.mean(bias['ssa']**2, axis=0))} # THIS IS MISSING FINE AND SSA
        true = {'βext':np.array(pxDct['trueExt']), 'βextFine':np.array(pxDct['trueExtFine']),
                'ssa':np.array(pxDct['trueSSA'])}
        return rmse, bias, true # each w/ keys: βext, βextFine, ssa

    def analyzeSim(self, wvlnthInd=0, modeCut=None, hghtCut=None, fineModesFwd=None, fineModesBck=None):
        """ Returns the RMSE and bias (defined below) from the simulation results
                wvlngthInd - the index of the wavelength to calculate stats for
                modeCut - fine/coarse seperation radius in um, currenltly only applied to rEff (None -> do not calculate error's modal dependence)
                hghtCut - PBL/FT seperation in meters (None -> do not calculate error's layer dependence)
                            if VarX_PBL in fwd keys (OSSE case), fwd PBL value for VarX will pulled from VarX_PBL and ignore hghtCut
                            in back OSSE case hghtCut argument is still used, although it could be pulled from 'pblh' key (see TODO notes below)
                fineModesFwd - [array-like] the indices of the fine modes in the foward calculation, set to None to use OSSE ..._Fine variables instead
                fineModesBck -  [array-like] the indices of the fine modes in the retrieval
                NOTE: this method generally assumes configuration (e.g. # of modes) is the same across all pixels
                TODO: we can do better than we do here in terms of fine mode parameters...
                    - we generally (always) use AOD weighting, which is not exact in many cases
                    - modeCut should be applied to more variables (e.g. spherical fraction)
                    - the PBL/FT variables should all be exact (formula taken from rtrvdDataSetPixels() in readOSSEnetCDF)
                """
        # check on input and available variables
        assert (self.rsltBck is not None) and (self.rsltFwd is not None), 'You must call loadSim() or runSim() before you can calculate statistics!'
        if type(self.rsltFwd) is dict: self.rsltFwd = [self.rsltFwd]
        assert type(self.rsltFwd) is list or type(self.rsltFwd) is np.ndarray, 'rsltFwd must be a list! Note that it was stored as a dict in older versions of the code.'
        fwdKys = self.rsltFwd[0].keys()
        bckKys = self.rsltBck[0].keys()
        msg = ' contains indices that are too high given the number of modes in '
        assert 'aodMode' not in fwdKys or fineModesFwd is None or self.rsltFwd[0]['aodMode'].shape[0] > max(fineModesFwd), 'fineModesFwd'+msg+'rsltFwd[aodMode]'
        assert 'aodMode' not in bckKys or fineModesBck is None or self.rsltBck[0]['aodMode'].shape[0] > max(fineModesBck), 'fineModesBck'+msg+'rsltBck[aodMode]'
        # define functions for calculating RMS and bias
        # rmsFun = lambda t,r: np.mean(r-t, axis=0) # formula for RMS output (true->t, retrieved->r)
        rmsFun = lambda t,r: np.sqrt(np.median((t-r)**2, axis=0)) # formula for RMS output (true->t, retrieved->r)
        biasFun = lambda t,r: r-t if r.ndim > 1 else np.atleast_2d(r-t).T # formula for bias output
        # variables we expect to see
        varsSpctrl = ['aod', 'aodMode', 'n', 'k', 'ssa', 'ssaMode', 'g', 'LidarRatio']
        varsMorph = ['rv', 'sigma', 'sph', 'rEffMode', 'rEffCalc', 'rEff', 'height']
        varsAodAvg = ['n', 'k'] # modal variables for which we will append aod weighted average RMSE and BIAS value at the FIRST element (expected to be spectral quantity)
        modalVars = ['rv', 'sigma', 'sph', 'aodMode', 'ssaMode','rEffMode', 'n', 'k'] # variables for which we find fine/coarse or FT/PBL errors seperately
        # calculate variables that weren't loaded (rEffMode)
        if 'rEffMode' not in self.rsltFwd[0] and 'rv' in self.rsltFwd[0]:
            for rf in self.rsltFwd: rf['rEffMode'] = self.ReffMode(rf)
        if 'rEffMode' not in self.rsltBck[0] and 'rv' in self.rsltBck[0]:
            for rb in self.rsltBck: rb['rEffMode'] = self.ReffMode(rb)
#         if hghtCut is None and 'pblh' in self.rsltFwd[0]: hghtCut = [rd['pblh'] for rd in self.rsltFwd] TODO: This won't work until hghtWghtedAvg can take in an array
        varsSpctrl = [z for z in varsSpctrl if z in fwdKys and z in bckKys] # check that the variable is used in current configuration
        varsMorph = [z for z in varsMorph if z in fwdKys and z in bckKys]
        # loop through varsSpctrl and varsMorph calcualted RMS and bias
        rmsErr = dict()
        bias = dict()
        trueOut = dict()
        for av in varsSpctrl+varsMorph:
            rtrvd = np.array([rb[av] for rb in self.rsltBck])
            true = np.array([rf[av] for rf in self.rsltFwd])
            if av in varsSpctrl:
                rtrvd = rtrvd[...,wvlnthInd]
                true = true[...,wvlnthInd]
            if rtrvd.ndim==1: rtrvd = np.expand_dims(rtrvd,1) # we want [ipix, imode], we add a singleton dimension if only one mode/nonmodal
            if true.ndim==1: true = np.expand_dims(true,1) # we want [ipix, imode], we add a singleton dimension if only one mode/nonmodal
            if hghtCut and av in modalVars and (av+'_PBL' in fwdKys or 'aodMode' in fwdKys): # calculate vertical dependent RMS/BIAS [PBL, FT*]
                if av+'_PBL' in fwdKys:
                    trueBilayer = self.getStateVals(av+'_PBL', self.rsltFwd, varsSpctrl, wvlnthInd)
                    pblOnly = av+'_FT' not in fwdKys
                    if not pblOnly:
                        trueFT = self.getStateVals(av+'_FT', self.rsltFwd, varsSpctrl, wvlnthInd)
                        trueBilayer = np.block([[trueBilayer],[trueFT]]).T # trueBilayer[pixel,mode]
                    rtrvdBilayer = self.hghtWghtedAvg(rtrvd, self.rsltBck, wvlnthInd, hghtCut, av, pblOnly=pblOnly)
                else: # 'aodMode' in self.rsltFwd[0]
                    trueBilayer = self.hghtWghtedAvg(true, self.rsltFwd, wvlnthInd, hghtCut, av)
                    rtrvdBilayer = self.hghtWghtedAvg(rtrvd, self.rsltBck, wvlnthInd, hghtCut, av)
                rmsErr[av+'_PBLFT'] = rmsFun(trueBilayer, rtrvdBilayer) # PBL is 1st ind, FT (not total column!) is 2nd
                bias[av+'_PBLFT'] = biasFun(trueBilayer, rtrvdBilayer)
                trueOut[av+'_PBLFT'] = trueBilayer
            if fineModesBck is not None and av in modalVars: # calculate fine mode dependent RMS/BIAS
                fineCalculated = False
                if av+'_fine' in fwdKys and fineModesFwd is None and 'aodMode' in bckKys: # we have OSSE outputs, currently user provided fineModesFwd trumps this though
                    trueFine = self.getStateVals(av+'_fine', self.rsltFwd, varsSpctrl, wvlnthInd)
                    rtrvdFine = self.getStateVals(av, self.rsltBck, varsSpctrl, wvlnthInd, fineModesBck)
                    fineCalculated = True
                elif fineModesFwd is not None and 'aodMode' in fwdKys and 'aodMode' in bckKys: # user provided fwd and bck fine mode indices
                    trueFine = self.getStateVals(av, self.rsltFwd, varsSpctrl, wvlnthInd, fineModesFwd)
                    rtrvdFine = self.getStateVals(av, self.rsltBck, varsSpctrl, wvlnthInd, fineModesBck)
                    fineCalculated = True
                elif fineModesFwd is not None: # something went wrong...
                    assert False, 'If fineModeFwd and fineModeBck are provided aodMode must be present in rsltsBck and rsltsFwd!'
                if fineCalculated:
                    rmsErr[av+'_fine'] = rmsFun(trueFine, rtrvdFine) # PBL is 1st ind, FT (not total column!) is 2nd
                    bias[av+'_fine'] =  biasFun(trueFine, rtrvdFine)
                    trueOut[av+'_fine'] =  trueFine
            if av in varsAodAvg: # calculate the total, mode AOD weighted value of the variable (likely just CRI) -> [total, mode1, mode2,...]
                rtrvd = np.hstack([self.τWghtedAvg(rtrvd, self.rsltBck, wvlnthInd), rtrvd])
                true = np.hstack([self.τWghtedAvg(true, self.rsltFwd, wvlnthInd), true])
            if true.shape[1] == rtrvd.shape[1]: # truth and retrieved modes can be paired one-to-one
                rmsErr[av] = rmsFun(true, rtrvd) # BUG: fineModesFwd and fineModesBck and not taken into accoutn here, really we just shouldn't return n or k with more than one mode (we have n(k)_fine now)
                bias[av] = biasFun(true, rtrvd)
                trueOut[av] = true
            elif av in varsAodAvg and 'aodMode' in fwdKys and 'aodMode' in bckKys: # we at least know the first, total elements of CRI correspond to each other
                rmsErr[av] = rmsFun(true[:,0], rtrvd[:,0])
                bias[av] = biasFun(true[:,0], rtrvd[:,0])
                trueOut[av] = true[:,0]
            if modeCut: # calculate rEff, could be abstracted into above code but tricky b/c volume weighted mean will not give exactly correct results (code below is exact)
                rtrvd = np.squeeze([self.ReffMode(rb, modeCut) for rb in self.rsltBck])
                true = np.squeeze([self.ReffMode(rf, modeCut) for rf in self.rsltFwd])
                modeCut_nm = 1000*modeCut
                rmsErr['rEff_sub%dnm' % modeCut_nm] = rmsFun(true, rtrvd)
                bias['rEff_sub%dnm' % modeCut_nm] = biasFun(true, rtrvd)
                trueOut['rEff_sub%dnm' % modeCut_nm] = true
                # TODO: add sph fraction here? It would use volWghtedAvg, I think in its exact current form
            if av in rmsErr: rmsErr[av] = np.atleast_1d(rmsErr[av]) # HACK: n was coming back as scalar in some cases, we should do this right though
        return rmsErr, bias, trueOut

    def getStateVals(self, av, rslts, varsSpctrl, wvlnthInd, modeInd=None):
        assert av in rslts[0], 'Variable %s was not found in the rslts dictionary!' % av
        stateVals = np.array([rf[av] for rf in rslts])
        if av.replace('_PBL','').replace('_FT','').replace('_fine','').replace('_coarse','') in varsSpctrl:
            stateVals = stateVals[...,wvlnthInd]
        if modeInd is not None and 'aod' in av: # we will sum multiple fine modes to get total fine mode AOD
            stateVals = np.expand_dims(stateVals[:, modeInd].sum(axis=1),1)
        elif modeInd is not None: # we will perform AOD weighted averaging from all fine modes of an intensive property
            stateVals = self.τWghtedAvg(stateVals[:, modeInd], rslts, wvlnthInd, modeInd)
        return stateVals

    def τWghtedAvg(self, val, rslts, wvlnthInd, modeInd=slice(None)):
        avgVal = np.full(len(rslts), np.nan)
        if (val.shape[1]==1 and slice(None)==modeInd) or 'aodMode' not in rslts[0]: # there was only one mode to start OR can't calculate b/c we don't have aodMode
            return np.zeros([val.shape[0], 0]) # return an empty array
        for i,rslt in enumerate(rslts):
            ttlSum = np.sum(val[i]*rslt['aodMode'][modeInd,wvlnthInd])
            normDenom = np.sum(rslt['aodMode'][modeInd,wvlnthInd])
            avgVal[i] = ttlSum/normDenom
        return np.expand_dims(avgVal, 1)

    def addProfFromGuassian(self, rslts):
        rsltRange = np.linspace(1, 2e4, 1e3)
        for rslt in rslts:
            Nmodes = len(rslt['height'])
            rslt['range'] = np.empty([Nmodes,len(rsltRange)])
            rslt['βext'] = np.empty([Nmodes,len(rsltRange)])
            for i, (μ,σ) in enumerate(zip(rslt['height'], rslt['heightStd'])): # loop over modes
                rslt['range'][i,:] = rsltRange
                guasDist = norm(loc=μ, scale=σ)
                rslt['βext'][i,:] = guasDist.pdf(rsltRange)

    def hghtWghtedAvg(self, val, rslts, wvlnthInd, hghtCut, av, pblOnly=False):
        """ quantities in val could correspond to: av { ['rv', 'sigma', 'sph', 'aodMode', 'ssaMode','n','k']
            ω=Σβh/Σαh => ω=Σωh*αh/Σαh, i.e. aod weighting below is exact for SSA
            sph is vol weighted and also exact
            there is no write answer for rv,σ,n and k so they are AOD weighted
            lidar ratio is exact, calculated by: S=τ/F11[180] & F11h[180]=τh/Sh => S=Στh/Σ(τh/Sh)
            --- reff vertical averaging (reff_h=Rh, # conc.=Nh, vol_conc.=Vh, height_ind=h) ---
            reff = ∫ ΣNh*r^3*dr/∫ ΣNh*r^2*dr = 3/4/π*ΣVh/∫ Σnh*r^2*dr
            Rh = (3/4/π*Vh)/∫ nh*r^2*dr -> reff = 3/4/π*ΣVh/(Σ(3/4/π*Vh)/Rh) = ΣVh/Σ(Vh/Rh)
        """
        assert av not in 'gMode', 'We dont have a wghtVals setting for asymetry parameter (i.e. ssaMode*aodMode)!'
        if 'range' not in rslts[0]: # this isn't lidar, we need to calucate the profiles
            self.addProfFromGuassian(rslts)
        if av in ['aod']:
            wghtFun = lambda v,w: np.sum(w)
        elif av in ['rEffMode', 'LidarRatio']:
            wghtFun = lambda v,w: np.sum(w)/np.sum(w/v)
        else: # SSA, rv, σ, n, k, gMode
            wghtFun = lambda v,w: np.sum(v*w)/np.sum(w)
        Bilayer = np.full([len(rslts), 2-pblOnly], np.nan)
        if np.any(rslts[0]['range']==hghtCut): warnings.warn('hghtCut fell exactly on one of the vertical bins, this bin will be excluded from the weighting.')
        for i,rslt in enumerate(rslts):
            if av in ['rEffMode', 'sph']: # weighting based on volume
                wghtVals = rslt['vol']
            else:  # weighting based on optical thickness
                wghtVals = rslt['aodMode'][:,wvlnthInd]
            wghtsPBL = []
            for β,h,τ in zip(rslt['βext'], rslt['range'], wghtVals): # loop over modes
                wghtsPBL.append(β[h < hghtCut].sum()/β.sum()*τ) # this is τ contribution of each mode below hghtCut
            valPBL = wghtFun(val[i], wghtsPBL)
            if not pblOnly:
                wghtsFT = []
                for β,h,τ in zip(rslt['βext'], rslt['range'], wghtVals):
                    wghtsFT.append(β[h > hghtCut].sum()/β.sum()*τ)
                valFT = wghtFun(val[i], wghtsFT)
                Bilayer[i] = [valPBL, valFT]
            else:
                Bilayer[i] = [valPBL]
        return Bilayer

    def ReffMode(self, rs, modeCut=None):
        if 'rv' in rs and 'sigma' in rs:
            if not modeCut:
                rv_dndr = rs['rv']/np.exp(3*rs['sigma']**2)
                return rv_dndr*np.exp(5/2*rs['sigma']**2) # eq. 60 from Grainger's "Useful Formulae for Aerosol Size Distributions"
            Vfc = self.volWghtedAvg(None, [rs], modeCut)
            Amode = rs['vol']/rs['rv']*np.exp(rs['sigma']**2/2) # convert N to rv and then ratio 1st and 4th rows of Table 1 of Grainger's "Useful Formulae for Aerosol Size Distributions"
            Afc = self.volWghtedAvg(None, [rs], modeCut, Amode)
            return Vfc/Afc # NOTE: ostensibly this should be Vfc/Afc/3 but we ommited the factor of 3 from Amode as well
        elif 'dVdlnr' in rs and 'r' in rs:
            if not modeCut: return ms.effRadius(rs['r'], rs['dVdlnr'])
            dvdlnr = np.atleast_2d(rs['dVdlnr'])
            radii = np.atleast_2d(rs['r'])[0,:] # we are assuming every mode has the same radii bins here...
            for i,(vol,psd) in enumerate(zip(np.atleast_1d(rs['vol']), dvdlnr)): # loop through modes
                dvdlnr[i] = psd*vol/np.trapz(psd/radii, x=radii) # scale to real units (i.e. "unnormalize" it)
            fnCrsReff = np.empty(2)
            for i,vldInd in enumerate([radii<=modeCut, radii>modeCut]): # loop twice, calculating fine then coarse result
                fnCrsReff[i] = ms.effRadius(radii[vldInd], dvdlnr[:,vldInd].sum(axis=0)) # sum over modes to obtain total PSD, then calculate rEff
            return fnCrsReff
        else:
            assert False, 'Can not calculate effective radius without either rv & σ OR dVdlnr & r!'

    def volWghtedAvg(self, val, rslts, modeCut, vol=None, fineOnly=False):
        N = 200 # number of radii bins
        Bimode = np.full([len(rslts), 2-fineOnly], np.nan)
        for i,rslt in enumerate(rslts): # loop over each pixel/time
            sigma4 = np.exp(4*rslt['sigma'].max())
            minVal = rslt['rv'].min()/sigma4
            maxVal = rslt['rv'].max()*sigma4
            lower = [minVal] if fineOnly else [minVal, modeCut]
            upper = [modeCut] if fineOnly else [modeCut, maxVal]
            crsWght = [] # this will be [upr/lwr(N=2), mode]
            for upr, lwr, in zip(upper,lower):
                r = np.logspace(np.log10(lwr),np.log10(upr),N)
                crsWght.append([np.trapz(ms.logNormal(mu, σ, r)[0],r) for mu,σ in zip(rslt['rv'],rslt['sigma'])]) # integrated r 0->inf this will sum to unity
            if not np.isclose(np.sum(crsWght, axis=0), 1, rtol=0.001).all():
                warnings.warn('The sum of the crsWght values across all modes was greater than 0.1% from unity') # probably need to adjust N or sigma4 above
            if val is None: # we just want volume (or area if vol arugment contains area) concentration of each mode
                if vol is None:
                    crsWght = np.array(crsWght)*rslt['vol']
                else:
                    crsWght = np.array(crsWght)*vol
                Bimode[i] = np.sum(crsWght, axis=1)
            else:
                Bimode[i] = np.sum(crsWght*val[i], axis=1)
        return Bimode
