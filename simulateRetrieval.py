#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
import pickle
import re
import runGRASP as rg

class simulation(object):
    def __init__(self, nowPix=None, addError=None):
        if nowPix is None: return
        assert np.all(np.diff([[mt for mt in mv['meas_type']] for mv in nowPix.measVals])>0), 'nowPix.measVals[l][\'meas_type\'] must be in ascending order at each l'
        self.nowPix = nowPix
        self.addError = addError
        self.measNm = np.array([[self.msType2msName(mt) for mt in mv['meas_type']] for mv in self.nowPix.measVals])
        self.nbvm = np.array([mv['nbvm'] for mv in nowPix.measVals])
        self.rsltBck = None
        self.rsltFwd = None
        
    def msType2msName(self, msTypeInt):
        msType = str(msTypeInt)
        msType = msType.replace('41','I')
        msType = msType.replace('42','Q')
        msType = msType.replace('43','U')
        assert re.match('^[A-z]+$', msType), 'Could not match msType %s to a valid measurment name!' % msType
        return msType
    
    def runSim(self, fwdModelYAMLpath, bckYAMLpath, Nsims=100, maxCPU=4, binPathGRASP=None, savePath=None, intrnlFileGRASP=None):
        assert not self.nowPix is None, 'A dummy pixel (nowPix) and error function (addError) are needed in order to run the simulation.' 
        # RUN THE FOWARD MODEL
        gObjFwd = rg.graspRun(fwdModelYAMLpath)
        gObjFwd.addPix(self.nowPix)
        gObjFwd.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=intrnlFileGRASP)
        self.rsltFwd = gObjFwd.readOutput()[0]
        # ADD NOISE AND PERFORM RETRIEVALS
        gObjBck = rg.graspRun(bckYAMLpath)
        for i in range(Nsims):
            for l, msDct in enumerate(self.nowPix.measVals):
                edgInd = np.r_[0, np.cumsum(self.nbvm[l,:])]
                msDct['measurements'] = copy.copy(msDct['measurements']) # we are about to write to this
                msDct['measurements'] = self.addError(l, self.rsltFwd, self.measNm[l,:], edgInd)
            self.nowPix.dtNm = copy.copy(self.nowPix.dtNm+1) # otherwise GRASP will whine
            gObjBck.addPix(self.nowPix)
        gDB = rg.graspDB(gObjBck, maxCPU)
        self.rsltBck = gDB.processData(maxCPU, binPathGRASP, krnlPathGRASP=intrnlFileGRASP)
        # SAVE RESULTS
        if savePath:
            with open(savePath, 'wb') as f:
                pickle.dump(self.rsltBck.tolist()+[self.rsltFwd], f, pickle.HIGHEST_PROTOCOL)

    def loadSim(self, picklePath):
        gDB = rg.graspDB()
        loadData = gDB.loadResults(picklePath)
        self.rsltBck = loadData[:-1]
        self.rsltFwd = loadData[-1]

    def analyzeSim(self, wvlnthInd=0):
        varsSpctrl = ['aod', 'n', 'k', 'ssa']
        varsMorph = ['rv', 'sigma', 'sph', 'rEffCalc']
        rmsErr = dict()
        meanBias = dict()
        assert (not self.rsltBck is None) and self.rsltFwd, 'You must call loadSim() or runSim() before you can calculate statistics!'
        for av in varsSpctrl+varsMorph:
            if av in varsSpctrl:
                rtrvd = [rs[av][...,wvlnthInd] for rs in self.rsltBck] # wavelength is always the last dimension
                true = self.rsltFwd[av][...,wvlnthInd]
            else:
                rtrvd = [rs[av] for rs in self.rsltBck]
                true = self.rsltFwd[av]
            rmsErr[av] = np.sqrt(np.mean((true-rtrvd)**2, axis=0))
            meanBias[av] = np.mean(true-rtrvd, axis=0)
        return rmsErr, meanBias
    
    
        
        
        
