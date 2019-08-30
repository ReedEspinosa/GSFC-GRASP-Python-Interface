#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:40:41 2019

@author: wrespino
"""

import numpy as np
import copy
import pickle
import runGRASP as rg

class simulation(object):
    def __init__(self, nowPix, addError, measNm):
        self.nowPix = nowPix
        self.addError = addError
        self.measNm = measNm
        self.nbvm = np.array([mv['nbvm'] for mv in nowPix.measVals])
        self.rsltBck = None
        self.rsltFwd = None
        self.gDB = rg.graspDB()
    
    def runSim(self, fwdModelYAMLpath, bckYAMLpath, Nsims=100, savePath=None):
        # RUN THE FOWARD MODEL
        gObjFwd = rg.graspRun(fwdModelYAMLpath)
        gObjFwd.addPix(self.nowPix)
        gObjFwd.runGRASP()
        self.rsltFwd = gObjFwd.readOutput()
        # ADD NOISE AND PERFORM RETRIEVALS
        gObjBck = rg.graspRun(bckYAMLpath)
        for i in range(Nsims):
            self.nowPix.dtNm = copy.copy(self.nowPix.dtNm)
            for l, msDct in enumerate(self.nowPix.measVals):
                edgInd = np.r_[0, np.cumsum(self.nbvm[l,:])]
                msDct['measurements'] = copy.copy(msDct['measurements']) # we are going to write to this
                for i in range(len(self.nbvm[l,:])):
                    fwdSim = self.rsltFwd[0]['fit_'+self.measNm[l,i]][:,l]
                    fwdSim = copy.copy(self.addError(fwdSim, self.measNm[l,i]))
                    msDct['measurements'][edgInd[i]:edgInd[i+1]] = fwdSim
            self.nowPix.dtNm = self.nowPix.dtNm+1 # otherwise GRASP will whine
            gObjBck.addPix(self.nowPix)
        gObjBck.runGRASP()
        self.rsltBck = gObjBck.readOutput()
        # SAVE RESULTS
        if savePath:
            with open(savePath, 'wb') as f:
                pickle.dump(self.rsltBck+self.rsltFwd, f, pickle.HIGHEST_PROTOCOL)

    def loadSim(self, picklePath):
        loadData = self.gDB.loadResults(picklePath)
        self.rsltBck = loadData[:-1]
        self.rsltFwd = loadData[-1]

    def analyzeSim(self):
        assert self.rsltBck and self.rsltFwd, 'You must call loadSim() or runSim() before you can calculate statistics!"
        print('NOT COMPLETE')        