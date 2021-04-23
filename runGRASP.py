#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import os.path
import warnings
import re
import time
import pickle
import copy
from csv import writer as csv_writer
from datetime import datetime as dt  # we want datetime.datetime
from datetime import timedelta
from shutil import copyfile
from subprocess import Popen,PIPE
import pandas as pd
from netCDF4 import Dataset
from scipy.stats import gaussian_kde
import numpy as np
import yaml # may require `conda install pyyaml`
import miscFunctions as mf

try:
    import matplotlib.pyplot as plt
    PLOT_LOADED = True
except ImportError:
    PLOT_LOADED = False


class graspDB():
    def __init__(self, graspRunObjs=[], maxCPU=None, maxT=None):
        """
        The graspDB class is designed to run many instances of GRASP in parallel/sequence (depending on maxCPU and maxT below).
        It also contains a variety of (rather old) plotting functions to apply to many grasp results
        INPUTS: graspRunObjs - what it sounds like, a list of grasp run objects
                maxCPU - the maximum number of GRASP process to run at any given time (each process consumes one core); None for no max
                maxT - the maximum number of pixels to include in a single GRASP process; None for no max (not used if graspRunObjs is list)
        """
        self.maxCPU = maxCPU
        if type(graspRunObjs) is list:
            self.grObjs = graspRunObjs
        elif 'graspRun' in type(graspRunObjs).__name__:
            assert maxCPU or maxT, 'maxCPU or maxT must be provided to subdivide the graspRun! If you want to process a single run sequentially pass it as a single element list.'
            self.grObjs = []
            Npix = len(graspRunObjs.pixels)
            if maxCPU and maxT:
                grspChnkSz = int(min(int(np.ceil(Npix/maxCPU)), maxT))
            else:
                grspChnkSz = int(maxT if maxT else int(np.ceil(Npix/maxCPU)))
            strtInds = np.r_[0:Npix:grspChnkSz]
            for strtInd in strtInds:
                gObj = graspRun(graspRunObjs.yamlObj.YAMLpath, graspRunObjs.orbHght/1e3, releaseYAML=graspRunObjs.releaseYAML) # not specifying dirGRASP so a new temp dir is created
                endInd = min(strtInd+grspChnkSz, Npix)
                for ind in range(strtInd, endInd):
                    gObj.addPix(graspRunObjs.pixels[ind])
#                 HACK to test RT accuracy
#                 valueMap = np.r_[23:51]
#                 fieldTest = 'retrieval.radiative_transfer.simulating_observation.number_of_guassian_quadratures_for_fourier_expansion_coefficients'
#                 gObj.yamlObj.access(fieldTest, valueMap[strtInd])
                self.grObjs.append(gObj)
        else:
            assert not graspRunObjs, 'graspRunObj must be either a list or graspRun object!'

    def processData(self, maxCPUs=None, binPathGRASP=None, savePath=False, krnlPathGRASP=None, nodesSLURM=0, rndGuess=False):
        if not maxCPUs:
            maxCPUs = self.maxCPU if self.maxCPU else 2
        usedDirs = []
        t0 = time.time()
        for grObj in self.grObjs:
            if grObj.dirGRASP: # If not it will be deduced later when writing SDATA
                assert (grObj.dirGRASP not in usedDirs), "Each graspRun instance must use a unique directory!"
                usedDirs.append(grObj.dirGRASP)
            grObj.writeSDATA()
        if nodesSLURM==0:
            i = 0
            Nobjs = len(self.grObjs)
            pObjs = []
            while i < Nobjs:
                if sum([pObj.poll() is None for pObj in pObjs]) < maxCPUs:
                    print('Starting a new thread for graspRun index %d/%d' % (i+1, Nobjs))
                    if rndGuess: self.grObjs[i].yamlObj.scrambleInitialGuess(rndGuess)
                    pObjs.append(self.grObjs[i].runGRASP(True, binPathGRASP, krnlPathGRASP))
                    i += 1
                time.sleep(0.1)
            while any([pObj.poll() is None for pObj in pObjs]): time.sleep(0.1)
            failedRuns = np.array([pObj.returncode for pObj in pObjs])>0
            for flRn in failedRuns.nonzero()[0]:
                print(' !!! Exit code %d in: %s' % (pObjs[flRn].returncode, self.grObjs[flRn].dirGRASP))
                for line in iter(pObjs[flRn].stdout.readline, b''):
                    errMtch = re.match('^ERROR:.*$', line.decode("utf-8"))
                    if errMtch is not None: print(errMtch.group(0))
            [pObj.stdout.close() for pObj in pObjs]
        else:
            # N steps through self.grObjs maxCPU's at a time
            # write runGRASP_N.sh using new dryRun option on runGRASP foreach maxCPU self.grObjs
            #   (dry run option would just return the command grasp settings...yml)
            # edit slurmTest2.sh to call runGRASP_N.sh with --array=1-N
            # set approriate flags so that grObj.readOutput() can be called without error
            # failedRuns = [bool with true for grObjs that failed to run with exit code 0]
            assert False, 'DIRECT USE OF SLURM IS NOT YET SUPPORTED'
        self.rslts = []
#        [self.rslts.extend(grObj.readOutput()) for grObj in self.grObjs[~failedRuns]]
        [self.rslts.extend(self.grObjs[i].readOutput()) for i in np.nonzero(~failedRuns)[0]]
        dtSec = time.time() - t0
        print('%d pixels processed in %8.2f seconds (%5.2f pixels/second)' % (len(self.rslts), dtSec, len(self.rslts)/dtSec))
        if savePath:
            with open(savePath, 'wb') as f:
                pickle.dump(self.rslts, f, pickle.HIGHEST_PROTOCOL)
        self.rslts = np.array(self.rslts) # numpy lists indexed w/ only assignment (no copy) but prior code built for std. list
        return self.rslts

    def loadResults(self, loadPath):
        try:
            with open(loadPath, 'rb') as f:
                self.rslts = np.array(pickle.load(f))
            return self.rslts
        except EnvironmentError:
            warnings.warn('Could not load valid pickle data from %s.' % loadPath)
            return []

    def histPlot(self, VarNm, Ind=0, customAx=False, FS=14, rsltInds=slice(None),
                 pltLabel=False, clnLayout=True): # clnLayout==False produces some speed up
        assert PLOT_LOADED, 'matplotlib could not be loaded, plotting features unavailable.'
        VarVal = self.getVarValues(VarNm, Ind, rsltInds)
        VarVal = VarVal[~pd.isnull(VarVal)]
        assert VarVal.shape[0] > 0, 'Zero valid matchups were found!'
        if customAx: plt.sca(customAx)
        plt.hist(VarVal, bins='auto')
        plt.xlabel(self.getLabelStr(VarNm, Ind))
        plt.ylabel('frequency')
        self.plotCleanUp(pltLabel, clnLayout)

    def scatterPlot(self, xVarNm, yVarNm, xInd=0, yInd=0, cVarNm=False, cInd=0, customAx=False,
                    logScl=False, Rstats=False, one2oneScale=False, FS=14, rsltInds=slice(None),
                    pltLabel=False, clnLayout=True): # clnLayout==False produces some speed up
        assert PLOT_LOADED, 'matplotlib could not be loaded, plotting features unavailable.'
        xVarVal = self.getVarValues(xVarNm, xInd, rsltInds)
        yVarVal = self.getVarValues(yVarNm, yInd, rsltInds)
        zeroErrStr = 'Values must be greater than zero for log scale!'
        noValPntsErrstr = 'Zero valid matchups were found!'
        if not cVarNm: # color by density
            vldInd = ~np.any((pd.isnull(xVarVal),pd.isnull(yVarVal)), axis=0)
            assert np.any(vldInd), noValPntsErrstr
            xVarVal = xVarVal[vldInd]
            yVarVal = yVarVal[vldInd]
            assert (not logScl) or ((xVarVal > 0).all() and (yVarVal > 0).all()), zeroErrStr
            if type(xVarVal[0])==dt or type(yVarVal[0])==dt: # don't color datetimes by density
                clrVar = np.zeros(xVarVal.shape[0])
            else:
                xy = np.log(np.vstack([xVarVal,yVarVal])) if logScl else np.vstack([xVarVal,yVarVal])
                clrVar = gaussian_kde(xy)(xy)
                if 'aod' in xVarNm: clrVar = clrVar**0.25
        else:
            clrVar = self.getVarValues(cVarNm, cInd, rsltInds)
            vldInd = ~np.any((pd.isnull(xVarVal),pd.isnull(yVarVal),pd.isnull(clrVar)), axis=0)
#            vldInd = np.logical_and(vldInd, np.abs(clrVar)<np.nanpercentile(np.abs(clrVar),20.0)) # HACK to stretch color scale
            assert np.any(vldInd), noValPntsErrstr
            xVarVal = xVarVal[vldInd]
            yVarVal = yVarVal[vldInd]
            clrVar = clrVar[vldInd]
            assert (not logScl) or ((xVarVal > 0).all() and (yVarVal > 0).all()), zeroErrStr
        # GENERATE PLOTS
        if customAx: plt.sca(customAx)
        MS = 2 if xVarVal.shape[0] < 5000 else 1
        plt.scatter(xVarVal, yVarVal, c=clrVar, marker='.', s=MS, cmap='inferno')
        plt.xlabel(self.getLabelStr(xVarNm, xInd))
        plt.ylabel(self.getLabelStr(yVarNm, yInd))
        nonNumpy = not (type(xVarVal[0]).__module__ == np.__name__ and type(yVarVal[0]).__module__ == np.__name__)
        if Rstats and nonNumpy:
            warnings.warn('Rstats can not be calculated for non-numpy types')
            Rstats = False
        if one2oneScale and nonNumpy:
            warnings.warn('one2oneScale can not be used with non-numpy types')
            one2oneScale = False
        if Rstats:
            Rcoef = np.corrcoef(xVarVal, yVarVal)[0,1]
            RMSE = np.sqrt(np.mean((xVarVal - yVarVal)**2))
            bias = np.mean((yVarVal-xVarVal))
            textstr = 'N=%d\nR=%.3f\nRMS=%.3f\nbias=%.3f' % (len(xVarVal), Rcoef, RMSE, bias)
            tHnd = plt.annotate(textstr, xy=(0, 1), xytext=(4.5, -4.5), va='top', xycoords='axes fraction',
                                textcoords='offset points', color='b', FontSize=FS)
            tHnd.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='None'))
        if logScl:
            plt.yscale('log')
            plt.xscale('log')
        if one2oneScale:
            maxVal = np.max(np.r_[xVarVal,yVarVal])
            if logScl:
                minVal = np.min(np.r_[xVarVal,yVarVal])
                plt.plot(np.r_[minVal, maxVal], np.r_[minVal, maxVal], 'k')
                plt.xlim([minVal, maxVal])
                plt.ylim([minVal, maxVal])
            else:
                plt.plot(np.r_[-1, maxVal], np.r_[-1, maxVal], 'k')
                plt.xlim([-0.01, maxVal])
                plt.ylim([-0.01, maxVal])
        elif logScl: # there is a bug in matplotlib that screws up scale
            plt.xlim([xVarVal.min(), xVarVal.max()])
            plt.ylim([yVarVal.min(), yVarVal.max()])
        if cVarNm:
            clrHnd = plt.colorbar()
            clrHnd.set_label(self.getLabelStr(cVarNm, cInd))
        self.plotCleanUp(pltLabel, clnLayout)

    def diffPlot(self, xVarNm, yVarNm, xInd=0, yInd=0, customAx=False,
                 rsltInds=slice(None), FS=14, logSpaceBins=True, lambdaFuncEE=False, # lambdaFuncEE = lambda x: 0.03+0.1*x (DT C6 Ocean EE)
                 pltLabel=False, clnLayout=True): # clnLayout==False produces moderate speed up
        assert PLOT_LOADED, 'matplotlib could not be loaded, plotting features unavailable.'
        xVarVal = self.getVarValues(xVarNm, xInd, rsltInds)
        yVarVal = self.getVarValues(yVarNm, yInd, rsltInds)
        vldInd = ~np.any((pd.isnull(xVarVal),pd.isnull(yVarVal)), axis=0)
        assert np.any(vldInd), 'Zero valid matchups were found!'
        xVarVal = xVarVal[vldInd]
        yVarVal = yVarVal[vldInd]
        if logSpaceBins:
            binEdge = np.exp(np.histogram(np.log(xVarVal),bins='sturges')[1])
            binMid = np.sqrt(binEdge[1:]*binEdge[:-1])
        else:
            binEdge = np.histogram(xVarVal, bins='sturges')[1]
            binMid = (binEdge[1:]+binEdge[:-1])/2
        if 'aod' in xVarNm:
            binEdge = np.delete(binEdge, np.nonzero(binMid < 0.005)[0]+1)
            binMid = binMid[binMid>=0.005]
            binEdge = np.delete(binEdge, np.nonzero(binMid > 2.5)[0])
            binMid = binMid[binMid<=2.5]
        varDif = yVarVal-xVarVal
        varRng = np.zeros([binMid.shape[0],3]) # lower (16%), mean, upper (84%)
        for i in range(binMid.shape[0]):
            varDifNow = varDif[np.logical_and(xVarVal > binEdge[i], xVarVal <= binEdge[i+1])]
            if varDifNow.shape[0]==0:
                varRng[i,:] = np.nan
            else:
                varRng[i,0] = np.percentile(varDifNow, 16)
                varRng[i,1] = np.mean(varDifNow)
                varRng[i,2] = np.percentile(varDifNow, 84)
        binMid = binMid[~np.isnan(varRng[:,0])]
        varRng = varRng[~np.isnan(varRng[:,0]),:]
        if customAx: plt.sca(customAx)
        if logSpaceBins: plt.xscale('log')
        plt.plot([binEdge[0],binEdge[-1]], [0,0], 'k')
        if lambdaFuncEE:
            if logSpaceBins:
                x = np.logspace(np.log10(binEdge[0]), np.log10(binEdge[-1]), 1000)
            else:
                x = np.linspace(binEdge[0], binEdge[-1], 1000)
            y = lambdaFuncEE(x)
            plt.plot(x, y, '--', color=[0.5,0.5,0.5])
            plt.plot(x, -y, '--', color=[0.5,0.5,0.5])
        errBnds = np.abs(varRng[:,1].reshape(-1,1)-varRng[:,[0,2]]).T
        plt.errorbar(binMid, varRng[:,1], errBnds, ecolor='r', color='r', marker='s', linstyle=None)
        plt.xlim([binMid[0]/1.2, 1.2*binMid[-1]])
        plt.xlabel(self.getLabelStr(xVarNm, xInd))
        plt.ylabel('%s-%s' % (yVarNm, xVarNm))
        if lambdaFuncEE:
            Nval = xVarVal.shape[0]
            inEE = 100*np.sum(np.abs(varDif) < lambdaFuncEE(xVarVal))/Nval
            in2EE = 100*np.sum(np.abs(varDif) < 2*lambdaFuncEE(xVarVal))/Nval
            txtStr = 'N=%d\nwithin 1xEE: %4.1f%%\nwithin 2xEE: %4.1f%%' % (Nval, inEE, in2EE)
            b = lambdaFuncEE(0)
            m = lambdaFuncEE(1) - b
            if np.all(y==m*x+b): # safe to assume EE function is linear
                txtStr = txtStr + '\nEE=%.2g+%.2gτ' % (b, m)
            plt.annotate(txtStr, xy=(0, 1), xytext=(4.5, -4.5), va='top', xycoords='axes fraction',
                         textcoords='offset points', FontSize=FS, color='b')
            plt.ylim([np.min([pltY, 2*y.max()]) for pltY in plt.ylim()]) # confine ylim to twice max(EE)
        plt.ylim([-np.abs(plt.ylim()).max(), np.abs(plt.ylim()).max()]) # force zero line to middle
        self.plotCleanUp(pltLabel, clnLayout)

    def getVarValues(self, VarNm, fldIndRaw, rsltInds=slice(None)):
        assert hasattr(self, 'rslts'), 'You must run GRASP or load existing results before plotting.'
        fldInd = self.standardizeInd(VarNm, fldIndRaw)
        if np.any(fldInd==-1): # datetimes and lat/lons are scalars and not indexable
            if fldIndRaw!=0:
                warnings.warn('Ignoring index value %d for scalar %s' % (fldIndRaw, VarNm))
            return np.array([rslt[VarNm] for rslt in self.rslts[rsltInds]])
        return np.array([rslt[VarNm][tuple(fldInd)] for rslt in self.rslts[rsltInds]])

    def getLabelStr(self, VarNm, IndRaw):
        fldInd = self.standardizeInd(VarNm, IndRaw)
        adTxt = ''
        if type(self.rslts[0][VarNm]) == dt:
            adTxt = 'year, '
        elif np.all(fldInd!=-1):
            wvl = 0
            if VarNm=='aodDT' and 'lambdaDT' in self.rslts[0].keys():
                wvl = self.rslts[0]['lambdaDT'][fldInd[-1]]
            elif VarNm=='aodDB':
                wvl = self.rslts[0]['lambdaDB'][fldInd[-1]]
            elif np.isin(self.rslts[0]['lambda'].shape[0], self.rslts[0][VarNm].shape): # may trigger false label for some variables if Nwvl matches Nmodes or Nparams
                wvl = self.rslts[0]['lambda'][fldInd[-1]] # wvl is last ind; assume wavelengths are constant
            if wvl > 0: adTxt = adTxt + '%5.2g μm, ' % wvl
            if VarNm=='wtrSurf' or VarNm=='brdf' or VarNm=='bpdf':
                adTxt = adTxt + 'Param%d, ' % fldInd[0]
            elif not np.isscalar(self.rslts[0]['vol']) and self.rslts[0][VarNm].shape[0] == self.rslts[0]['vol'].shape[0]:
                adTxt = adTxt + 'Mode%d, ' % fldInd[0]
        if len(adTxt)==0:
            return VarNm
        else:
            return VarNm + ' (' + adTxt[0:-2] + ')'

    def standardizeInd(self, VarNm, IndRaw):
        assert (VarNm in self.rslts[0]), '%s not found in retrieval results dict' % VarNm
        if type(self.rslts[0][VarNm]) != np.ndarray: # no indexing for these...
            return np.r_[-1]
        Ind = np.array(IndRaw, ndmin=1)
        if self.rslts[0][VarNm].ndim < len(Ind):
            Ind = np.r_[0] if np.all(Ind==0) else Ind[Ind!=0]
        elif self.rslts[0][VarNm].ndim > len(Ind):
            if self.rslts[0][VarNm].shape[0]==1: Ind = np.r_[0, Ind]
            if self.rslts[0][VarNm].shape[-1]==1: Ind = np.r_[Ind, 0]
        assert self.rslts[0][VarNm].ndim==len(Ind), 'Number of indices (%d) does not match diminsions of %s (%d)' % (len(Ind),VarNm,self.rslts[0][VarNm].ndim)
        assert self.rslts[0][VarNm].shape[0] > Ind[0], '1st index %d is out of bounds for variable %s' % (Ind[0], VarNm)
        if len(Ind)==2:
            assert self.rslts[0][VarNm].shape[1] > Ind[1], '2nd index %d is out of bounds for variable %s' % (Ind[1], VarNm)
        return np.array(Ind)

    def plotCleanUp(self, pltLabel=False, clnLayout=True):
        assert PLOT_LOADED, 'matplotlib could not be loaded, plotting features unavailable.'
        if pltLabel:
            plt.suptitle(pltLabel)
            if clnLayout: plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        elif clnLayout:
            plt.tight_layout()


class graspRun():
    def __init__(self, pathYAML=None, orbHghtKM=700, dirGRASP=None, releaseYAML=False, verbose=True):
        """
        pathYAML – yaml data: posix path string, graspYAML object or None
            Note: type(pathYAML)==graspYAML –> create new instance of a graspYAML object, with a duplicated YAML file in dirGRASP (see below)
        dirGRASP – grasp working directory, None for new temp dir; this needs to point to directory with SDATA file if writeSDATA is not called
        releaseYAML – allow this class to adjust YAML to match SDATA (e.g. change number of wavelengths)
        """
        self.releaseYAML = releaseYAML # allow automated modification of YAML, all index of wavelength involved fields MUST cover every wavelength
        self.pathSDATA = False
        self.verbose = verbose
        self.orbHght = orbHghtKM*1000
        self.pixels = []
        self.pObj = False
        self.invRslt = dict()
        if pathYAML is None:
            assert not dirGRASP, 'You can not have a working directory (dirGRASP) without a YAML file (pathYAML)!'
            self.dirGRASP = None
            self.yamlObj = graspYAML()
            return
        self.dirGRASP = tempfile.mkdtemp() if not dirGRASP else dirGRASP # set working dir
        if type(pathYAML)==str:
            if os.path.dirname(pathYAML) == self.dirGRASP: # if YAML is in specified working dir
                self.yamlObj = graspYAML(pathYAML)
            else: # we copy YAML file to new directory
                newPathYAML = os.path.join(self.dirGRASP, os.path.basename(pathYAML))
                self.yamlObj = graspYAML(pathYAML, newPathYAML)
        elif type(pathYAML)==graspYAML:
            if pathYAML.dl is not None: pathYAML.writeYAML() # incase there are unsaved changes
            if releaseYAML: # copy so we don't overwrite orinal
                newPathYAML = os.path.join(self.dirGRASP, os.path.basename(pathYAML.YAMLpath))
                self.yamlObj = graspYAML(pathYAML.YAMLpath, newPathYAML)
            elif os.path.dirname(pathYAML.YAMLpath) == self.dirGRASP: # if YAML is in specified working dir:
                self.yamlObj = pathYAML # path YAML is actually a YAML object, not a path
            else: # we copy YAML file to the directory that will have the SDATA file [THIS WILL BREAK RUN WITH JUST YAML!!!]
                newPathYAML = os.path.join(self.dirGRASP, os.path.basename(pathYAML.YAMLpath))
                self.yamlObj = graspYAML(pathYAML.YAMLpath, newPathYAML)
        else:
            assert False, 'pathYAML should be None, a string or a yaml object!'
        if self.verbose: print('Working in %s' % self.dirGRASP)

    def addPix(self, newPixel): # this is called once for each pixel
        self.pixels.append(copy.deepcopy(newPixel)) # deepcopy need to prevent changing via original pixel object outside of graspRun object

    def writeSDATA(self, pathSDATA=None):
        if len(self.pixels) == 0:
            warnings.warn('You must call addPix() at least once before writting SDATA!')
            return False
        if pathSDATA:
            self.pathSDATA = pathSDATA
        else:
            assert self.yamlObj.YAMLpath, 'You must initialize graspRun with a YAML file to write SDATA'
            self.pathSDATA = os.path.join(self.dirGRASP, self.yamlObj.access('sdata_fn'))
            assert (self.pathSDATA), 'Failed to read SDATA filename from '+self.yamlObj.YAMLpath
        unqTimes = np.unique([pix.dtObj for pix in self.pixels])
        SDATAstr = self.genSDATAHead(unqTimes)
        for unqTime in unqTimes:
            pixInd = np.nonzero(unqTime == np.array([pix.dtObj for pix in self.pixels]))[0]
            SDATAstr += self.genCellHead(pixInd)
            for ind in pixInd:
                SDATAstr += self.pixels[ind].genString()
        with open(self.pathSDATA, 'w') as fid:
            fid.write(SDATAstr)
            fid.close()

    def runGRASP(self, parallel=False, binPathGRASP=None, krnlPathGRASP=None):
        if not binPathGRASP: binPathGRASP = '/usr/local/bin/grasp'
        if not self.pathSDATA:
            self.writeSDATA()
        if self.releaseYAML:
            self.yamlObj.adjustLambda(np.max([px.nwl for px in self.pixels]))
            Nbins = [len(np.unique(mv['thetav'])) for mv in self.pixels[0].measVals if np.all(mv['meas_type']<40)]
            if np.greater(Nbins,0).any(): # we have lidar data
                assert np.all([Nbins[0]==bn for bn in Nbins]), 'Expected same number of vertical bins at every λ, instead found '+str(Nbins)
                self.yamlObj.adjustVertBins(Nbins[0])
        if krnlPathGRASP: self.yamlObj.access('path_to_internal_files', krnlPathGRASP)
        self.pObj = Popen([binPathGRASP, self.yamlObj.YAMLpath], stdout=PIPE)
        if not parallel:
            print('Running GRASP...')
#            self.pObj.wait()
            self.pObj.communicate() # This seems to keep things from hanging if there is a lot of output...
            self.pObj.stdout.close()
            self.invRslt = self.readOutput() # Why store rsltDict only if not parallel? I guess to keep it from being stored twice in memory in graspDB case?
        return self.pObj # returns Popen object, (PopenObj.poll() is not None) == True when complete

    def readOutput(self, customOUT=None): # customOUT is full path of unrelated output file to read
        if customOUT is None and not self.pObj:
            warnings.warn('You must call runGRASP() before reading the output!')
            return False
        if not customOUT and self.pObj.poll() is None:
            warnings.warn('GRASP has not yet terminated, output can only be read after retrieval is complete.')
            return False
        assert (customOUT or self.yamlObj.access('stream_fn')), 'Failed to read stream filename from '+self.yamlObj.YAMLpath
        outputFN = customOUT if customOUT else os.path.join(self.dirGRASP, self.yamlObj.access('stream_fn'))
        try:
            with open(outputFN) as fid:
                contents = fid.readlines()
        except FileNotFoundError:
            msg = '%s could not be found, probably because GRASP crashed. \n   Returning empty list in place of output data...'
            warnings.warn(msg % outputFN)
            return []
        rsltAeroDict, wavelengths = self.parseOutAerosol(contents)
        if not rsltAeroDict:
            warnings.warn('Aerosol data could not be read from %s\n  Returning empty list in place of output data...' % outputFN)
            return []
        rsltSurfDict = self.parseOutSurface(contents, Nλ=len(wavelengths))
        rsltFitDict = self.parseOutFit(contents, wavelengths)
        rsltPMDict = self.parsePhaseMatrix(contents, wavelengths)
        rsltDict = []
        try:
            for aero, surf, fit, pm, aux in zip(rsltAeroDict, rsltSurfDict, rsltFitDict, rsltPMDict, self.AUX_dict):
                rsltDict.append({**aero, **surf, **fit, **pm, **aux})
        except AttributeError: # the user did not create AUX_dict
            for aero, surf, fit, pm in zip(rsltAeroDict, rsltSurfDict, rsltFitDict, rsltPMDict):
                rsltDict.append({**aero, **surf, **fit, **pm})
        self.calcAsymParam(rsltDict)
        return rsltDict

    def _findRslts(self, rsltDict=None, customOUT=None):
        assert not (rsltDict is not None and customOUT is not None), 'Only one of rsltDict or customOUT should be provided, not both!'
        if customOUT is not None:
            return self.readOutput(customOUT)
        elif rsltDict:
            return rsltDict
        elif len(self.invRslt)>0:
            return self.invRslt
        else:
            return self.readOutput()

    def singleScat2CSV(self, csvPath, pixInd=0, rsltDict=None, customOUT=None):
        """ This function will take grasp output and dump it in a netCDF file
            PM of pixInd (defualt 0) will be saved (only one pixel saved at a time)
            If resultDict is provided, this data will be written to netCDF.
            If customOUT is provided, data from GRASP output text file specified will be written to netCDF.
            If neither are provided the output text file associated with current instance will be written """
        rslt = self._findRslts(rsltDict, customOUT)[pixInd]
        pmΚeys = ['p11', 'p12', 'p22', 'p33','p34','p44']
        assert pmΚeys[0] in rslt, 'p11 key not found in results. Is your YAML output set to print the phase matrix?'
        Nmodes = rslt[pmΚeys[0]].shape[1]
        Nλ = len(rslt['lambda'])
        Npm = np.sum([key in pmΚeys for key in rslt.keys()])
        with open(csvPath, 'w', newline='') as csvfile:
            dataWriter = csv_writer(csvfile, delimiter=',', quotechar='|')
            # write 2 header rows TODO: page these with leading column for ext,sca and angle
            emptyStrList = ['' for _ in range(Npm*Nλ-1)]
            modeRow = np.hstack([['mode%d' % m] + emptyStrList for m in range(Nmodes)])
            dataWriter.writerow(np.r_[[''], modeRow])
            emptyStrList = ['' for _ in range(Npm-1)]
            λRow = np.hstack([['%5.3f um' % λ] + emptyStrList for λ in rslt['lambda']])
            dataWriter.writerow(np.r_[[''], np.tile(λRow, Nmodes)])
            # write ext/sca data
            extndVect = ['' for _ in range(Npm-2)]
            dataWriter.writerow(np.r_[[''], np.tile(['Bext (m^2/m^3)','Bsca (m^2/m^3)']+extndVect, Nmodes*Nλ)])
            dataVect = [''] # first column has angle and thus is empty for sca/ext
            for mode in range(Nmodes):
                for λind in range(Nλ):
                    dataVect.append(rslt['aodMode'][mode,λind]/rslt['vol'][mode]) # bext_vol
                    dataVect.append(rslt['aodMode'][mode,λind]*rslt['ssaMode'][mode,λind]/rslt['vol'][mode]) # bsca_vol
                    dataVect.extend(extndVect)
            dataWriter.writerow(dataVect)
            # write PM data
            dataWriter.writerow(np.r_[['angle'], np.tile(pmΚeys[0:Npm], Nmodes*Nλ)])
            for angInd, ang in enumerate(rslt['angle'][:,0,0]):
                dataVect = [ang]
                for mode in range(Nmodes):
                    for λind in range(Nλ):
                        for key in pmΚeys[0:Npm]:
                            dataVect.append(rslt[key][angInd,mode,λind]) # painfully inefficient, but surprisingly fast
                dataWriter.writerow(dataVect)

    def output2netCDF(self, nc4Path, rsltDict=None, customOUT=None, seaLevel=False):
        """ This function will take grasp output and dump it in a netCDF file
            If resultDict is provided, this data will be written to netCDF.
            If customOUT is provided, data from GRASP output text file specified will be written to netCDF.
            If neither are provided the output text file associated with current instance will be written.
            seaLevel=True should be used with caution, see assumed ROD and depol. below. """
        rsltDict = self._findRslts(rsltDict, customOUT)
        Nvis = np.unique(rsltDict[0]['vis'][:,0]).shape[0]
        Nfis = np.unique(rsltDict[0]['fis'][:,0]).shape[0]
        for vis,fis in zip(rsltDict[0]['vis'].T, rsltDict[0]['fis'].T):
            msg = 'All wavelengths must have the same number of %s angles!'
            assert Nvis == np.unique(vis).shape[0], msg % 'viewing zenith'
            assert Nfis == np.unique(fis).shape[0], msg % 'relative azimuth'
        # create netCDF4 data file
        with Dataset(nc4Path, 'w', format='NETCDF4') as root_grp:
            root_grp.description = 'Results of GRASP run'
            varHnds = dict()
            # add dimensions and write corresponding variables
            tName = 'pixNumber'
            root_grp.createDimension(tName, len(rsltDict))
            varHnds[tName] = root_grp.createVariable(tName, 'u2', (tName))
            varHnds[tName][:] = np.r_[0:len(rsltDict)]
            varHnds[tName].units = 'none'
            varHnds[tName].long_name = 'index of pixel'
            λName = 'wavelength'
            root_grp.createDimension(λName, len(rsltDict[0]['lambda']))
            varHnds[λName] = root_grp.createVariable(λName, 'f4', (λName))
            varHnds[λName][:] = rsltDict[0]['lambda']
            varHnds[λName].units = 'μm'
            varHnds[λName].long_name = 'wavelength of measurement'
            visName = 'viewing_zenith'
            root_grp.createDimension(visName, Nvis)
            varHnds[visName] = root_grp.createVariable(visName, 'f4', (visName))
            varHnds[visName][:] = np.sort(np.unique(rsltDict[0]['vis'][:,0]))
            varHnds[visName].units = 'degree'
            varHnds[visName].long_name = 'viewing zenith angle'
            fisName = 'relative_azimuth'
            root_grp.createDimension(fisName, Nfis)
            varHnds[fisName] = root_grp.createVariable(fisName, 'f4', (fisName))
            varHnds[fisName][:] = np.sort(np.unique(rsltDict[0]['fis'][:,0]))
            varHnds[fisName].units = 'degree'
            varHnds[fisName].long_name = 'relative azimuth angle (φ_solar - φ_viewing)'
            # write data variables
            for key in rsltDict[0].keys(): # loop over keys
                if 'fit' in key or 'sca_ang' in key:
                    if key.replace('fit_','') in ['I','Q','U']:
                        var = key.replace('fit_','')
                        varHnds[var] = root_grp.createVariable(var, 'f8', (tName, λName, visName, fisName))
                        varHnds[var].units = 'none'
                        varHnds[var].long_name = '%s at TOA' % var
                    elif key == 'sca_ang':
                        var = key
                        varHnds[var] = root_grp.createVariable(var, 'f4', (tName, λName, visName, fisName))
                        varHnds[var].units = 'degree'
                        varHnds[var].long_name = 'scattering angle'
                    else:
                        assert False, 'This function does not know how to handle the variable %s' % key
                    for ti, rslt in enumerate(rsltDict): # loop over pixels
                        for λi in range(len(varHnds[λName])): # loop over wavelengths
                            for θi,θ in enumerate(varHnds[visName][:]): # loop over viewing zeniths
                                for φi,φ in enumerate(varHnds[fisName][:]): # loop over realtive azimuths
                                    ind = np.logical_and(np.isclose(rslt['vis'][:,λi], θ),
                                                         np.isclose(rslt['fis'][:,λi], φ)).nonzero()[0]
                                    assert ind.shape[0] == 1, "%d values were found for %s at pixel# %d, λind=%d, θv=%4.1f, φ=%4.1f!" % (ind.shape[0],key,ti,λi,θ,φ)
                                    varHnds[var][ti,λi,θi,φi] = rslt[key][ind[0],λi]
                elif key=='brdf' and rsltDict[0]['brdf'].shape[0]==3: # probably RTLS parameters
                    for i,var in enumerate(['RTLS_ISO', 'RTLS_VOL', 'RTLS_GEO']): # loop over the three RTLS kernels
                        varHnds[var] = root_grp.createVariable(var, 'f8', (tName, λName))
                        varHnds[var].units = 'none'
                        varHnds[var][:,:] = np.array([rslt['brdf'][i,:] for rslt in rsltDict]) # loop over times, select all λ
                    varHnds['RTLS_ISO'].long_name = 'Isotropic kernel of the RTLS model'
                    varHnds['RTLS_VOL'].long_name = 'Volume kernel of the RTLS model (MAIAC_vol/MAIAC_iso)'
                    varHnds['RTLS_GEO'].long_name = 'Geometric kernel of the RTLS model (MAIAC_geo/MAIAC_iso)'
                elif key=='brdf' and rsltDict[0]['brdf'].ndim==1: # probably Maignan parameters
                    var = 'maignan_parameter'
                    varHnds[var] = root_grp.createVariable(var, 'f8', (tName, λName))
                    varHnds[var].units = 'none'
                    varHnds[var][:,:] = np.array([rslt['bpdf'] for rslt in rsltDict]) # loop over times, select all λ
                    varHnds[var].long_name = 'Value of the Maignan model (exp(-NDVI)*C_maignan)'
                elif key=='aod':
                    var = 'aod'
                    varHnds[var] = root_grp.createVariable(var, 'f8', (tName, λName))
                    varHnds[var].units = 'none'
                    varHnds[var][:,:] = np.array([rslt['aod'] for rslt in rsltDict]) # loop over times, select all λ
                    varHnds[var].long_name = 'aerosol optical depth'
            if seaLevel: # This is a little nasty, need to double check numbers below before using
                var = 'ROD'
                varHnds[var] = root_grp.createVariable(var, 'f8', (λName))
                varHnds[var].units = 'none'
                varHnds[var][:] = self.seaLevelROD(varHnds[λName][:])
                varHnds[var].long_name = 'Rayleigh Optical Depth'
                var = 'rayleigh_depol'
                varHnds[var] = root_grp.createVariable(var, 'f8', (λName))
                varHnds[var].units = 'none'
                varHnds[var][:] = 0.0295*np.ones(len(varHnds[λName][:]))
                varHnds[var].long_name = 'Rayleigh Depolarization Ratio'

    def seaLevelROD(self, λtarget):
        λ =   np.r_[0.3600, 0.3800, 0.4100, 0.5500, 0.6700, 0.8700, 1.5500, 1.6500]
        rod = np.r_[0.5612, 0.4474, 0.3259, 0.0973, 0.0436, 0.0152, 0.0015, 0.0012]
        assert λ.min()<=λtarget.min() and λ.max()>=λtarget.max(), 'λtarget falls outside the range of pre-programed values!'
        return np.interp(λtarget, λ, rod**-0.25)**-4

    def parseOutDateTime(self, contents):
        results = []
        ptrnDate = re.compile('^[ ]*Date[ ]*:[ ]+')
        ptrnTime = re.compile('^[ ]*Time[ ]*:[ ]+')
        ptrnLon = re.compile('^[ ]*Longitude[ ]*:[ ]+')
        ptrnLat = re.compile('^[ ]*Latitude[ ]*:[ ]+')
        i = 0
        LatLonLinesFnd = 0 # GRASP prints these twice, assume lat/lon are last lines
        while i < len(contents) and LatLonLinesFnd < 2:
            line = contents[i]
            if not ptrnDate.match(line) is None: # Date
                dtStrCln = line[ptrnDate.match(line).end():-1].split()
                dates_list = [dt.strptime(date, '%Y-%m-%d').date() for date in dtStrCln]
            if not ptrnTime.match(line) is None: # Time (should come after Date in output)
                dtStrCln = line[ptrnTime.match(line).end():-1].split()
                times_list = [dt.strptime(time, '%H:%M:%S').time() for time in dtStrCln]
                for j in range(len(times_list)):
                    dtNow = dt.combine(dates_list[j], times_list[j])
                    results.append(dict(datetime=dtNow))
            if not ptrnLon.match(line) is None: # longitude
                lonVals = np.array(line[ptrnLon.match(line).end():-1].split(), dtype=np.float)
                for k,lon in enumerate(lonVals):
                    results[k]['longitude'] = lon
                LatLonLinesFnd += 1
            if not ptrnLat.match(line) is None: # longitude
                latVals = np.array(line[ptrnLat.match(line).end():-1].split(), dtype=np.float)
                for k,lat in enumerate(latVals):
                    results[k]['latitude'] = lat
                LatLonLinesFnd += 1
            i += 1
        if not len(results[-1].keys())==3:
            warnings.warn('Failure reading date/lat/lon from GRASP output!')
            return []
        return results

    def parseOutAerosol(self, contents):
        results = self.parseOutDateTime(contents)
        if len(results)==0:
            return []
        ptrnPSD = re.compile('^[ ]*(Radius \(um\),)?[ ]*Size Distribution dV\/dlnr \(normalized')
        ptrnProfile = re.compile('^[ ]*Aerosol vertical profile \[1\/m\] for Particle component [0-9]+')
        ptrnRange = re.compile('^[ ]*Aerosol vertical profile altitudes \[m\] for Particle component [0-9]+')
        ptrnLN = re.compile('^[ ]*Parameters of lognormal SD')
        ptrnVol = re.compile('^[ ]*Aerosol volume concentration')
        ptrnSPH = re.compile('^[ ]*% of spherical particles')
        ptrnHGNT = re.compile('^[ ]*Aerosol profile mean height')
        ptrnHGNTSTD = re.compile('^[ ]*Aerosol profile standard deviation')
        ptrnAOD = re.compile('^[ ]*Wavelength \(um\),[ ]+(Total_AOD|AOD_Total)')
        ptrnAODmode = re.compile('^[ ]*Wavelength \(um\),[ ]+AOD_Particle_mode')
        ptrnSSA = re.compile('^[ ]*Wavelength \(um\),[ ]+(SSA_Total|Total_SSA)')
        ptrnLidar = re.compile('^[ ]*Wavelength \(um\),[ ]+Lidar[ ]*Ratio[ ]*\(Total\)')
        ptrnSSAmode = re.compile('^[ ]*Wavelength \(um\),[ ]+SSA_Particle_mode')
        ptrnRRI = re.compile('^[ ]*Wavelength \(um\), REAL Ref\. Index')
        ptrnIRI = re.compile('^[ ]*Wavelength \(um\), IMAG Ref\. Index')
        ptrnReff = re.compile('^[ ]*reff total[ ]*([0-9Ee.+\- ]+)[ ]*$') # this seems to have been removed in GRASP V0.8.2, atleast with >1 mode
        i = 0
        nsd = 0
        rngAndβextUnited = True
        while i < len(contents): # loop line by line, checking each one against the patterns above
            if not ptrnLN.match(contents[i]) is None: # lognormal PSD, these fields have unique form
                mtch = re.search('[ ]*rv \(um\):[ ]*', contents[i+1])
                rvArr = np.array(contents[i+1][mtch.end():-1].split(), dtype='float64')
                mtch = re.search('[ ]*ln\(sigma\):[ ]*', contents[i+2])
                sigArr = np.array(contents[i+2][mtch.end():-1].split(), dtype='float64')
                for k in range(len(results)):
                    results[k]['rv'] = np.append(results[k]['rv'], rvArr[k]) if 'rv' in results[k] else rvArr[k]
                    results[k]['sigma'] = np.append(results[k]['sigma'], sigArr[k]) if 'sigma' in results[k] else sigArr[k]
                i += 2
            if not ptrnReff.match(contents[i]) is None: # Reff, field has unique form
                Reffs = np.array(ptrnReff.match(contents[i]).group(1).split(), dtype='float64')
                for k,Reff in enumerate(Reffs):
                    results[k]['rEff'] = Reff
            self.parseMultiParamFld(contents, i, results, ptrnAOD, 'aod', 'lambda')
            self.parseMultiParamFld(contents, i, results, ptrnPSD, 'dVdlnr', 'r')
            if rngAndβextUnited:
                try: # sometimes GRASP adds range column before extinction profile (if not this expects one more column than is present, producing an index error)
                    self.parseMultiParamFld(contents, i, results, ptrnProfile, 'βext', 'range', colOffset=1)
                except (IndexError, AssertionError): # but other times range is a separate field... no obvious rhyme/reason
                    del results[0]['βext'] # we should have crashed while setting βext in the first pixel, no need to delete key in others
                    for rd in results: del rd['range'] # range should have been set in every pixel before the crash
                    i -= 1 # we need to parse this line again, using the correct arguments for parseMultiParamFld (below)
                    rngAndβextUnited = False # we fixed the issues, and we now no better than to try again
            else:
                self.parseMultiParamFld(contents, i, results, ptrnProfile, 'βext')
                self.parseMultiParamFld(contents, i, results, ptrnRange, 'range')
            self.parseMultiParamFld(contents, i, results, ptrnVol, 'vol')
            self.parseMultiParamFld(contents, i, results, ptrnSPH, 'sph')
            self.parseMultiParamFld(contents, i, results, ptrnHGNT, 'height')
            self.parseMultiParamFld(contents, i, results, ptrnHGNTSTD, 'heightStd')
            self.parseMultiParamFld(contents, i, results, ptrnAODmode, 'aodMode')
            self.parseMultiParamFld(contents, i, results, ptrnSSA, 'ssa')
            self.parseMultiParamFld(contents, i, results, ptrnLidar, 'LidarRatio')
            self.parseMultiParamFld(contents, i, results, ptrnSSAmode, 'ssaMode')
            self.parseMultiParamFld(contents, i, results, ptrnRRI, 'n')
            self.parseMultiParamFld(contents, i, results, ptrnIRI, 'k')
            i += 1
        if not results or 'lambda' not in results[0]:
            warnings.warn('Limited or no aerosol data found, returning incomplete dictionary...')
            return results
        wavelengths = np.atleast_1d(results[0]['lambda'])
        if 'aodMode' in results[0]:
            Nwvlth = 1 if np.isscalar(results[0]['aod']) else results[0]['aod'].shape[0]
            nsd = int(results[0]['aodMode'].shape[0]/Nwvlth)
            for rs in results: # seperate aerosol modes
                rs['r'] = rs['r'].reshape(nsd,-1)
                rs['dVdlnr'] = rs['dVdlnr'].reshape(nsd,-1)
                for key in [k for k in ['aodMode','ssaMode','n','k'] if k in rs]:
                    if rs[key].shape[-1] == nsd*Nwvlth:
                        rs[key] = rs[key].reshape(nsd,-1) # we double check that -1 -> Nwvlth on next line
                    assert rs[key].shape[-1]==Nwvlth, 'Length of the last dimension of %s was %d, not matching Nλ=%d' % (key, rs[key].shape[-1], Nwvlth)
                for λflatKey in [k for k in ['n','k'] if k in rs.keys()]: # check if spectrally flat RI values used
                    for mode in rs[λflatKey]: mode[mode==0] = mode[0] # fill zero values with first value
                if 'βext' in rs:
                    rs['range'] = rs['range'].reshape(nsd,-1)
                    rs['βext'] = rs['βext'].reshape(nsd,-1)
                    βprfl = rs['βext'].sum(axis=0)
                    rng = rs['range'][0]
                    rs['height'] = np.trapz(βprfl*rng, rng)/np.trapz(βprfl, rng) # extinction weighted mean height
                    λ550Ind = np.argmin(np.abs(rs['lambda']-0.55))
                    for mode in range(nsd): # scale βext to 1/Mm at λ=550nm (or next closest λ)
                        AOD = rs['aodMode'][mode, λ550Ind]
                        rs['βext'][mode,:] = 1e6*mf.norm2absExtProf(rs['βext'][mode,:], rs['range'][mode,:], AOD)
        if ('r' in results[0]) and np.all(results[0]['r'][0]==results[0]['r']): # check if all r value are same at all lambda, may remove this condition later but makes logic much more complicated
            for rs in results:
                dvdlnr = (rs['dVdlnr']*np.atleast_2d(rs['vol']).T).sum(axis=0)
                rs['rEffCalc'] = (mf.effRadius(rs['r'][0], dvdlnr))
        return results, wavelengths

    def parseOutSurface(self, contents, Nλ=None):
        results = self.parseOutDateTime(contents)
        ptrnALB = re.compile('^[ ]*Wavelength \(um\),[ ]+Surface ALBEDO')
        ptrnBRDF = re.compile('^[ ]*Wavelength \(um\),[ ]+BRDF parameters')
        ptrnBPDF = re.compile('^[ ]*Wavelength \(um\),[ ]+BPDF parameters')
        ptrnWater = re.compile('^[ ]*Wavelength \(um\),[ ]+Water surface parameters')
        i = 0
        while i < len(contents):
            self.parseMultiParamFld(contents, i, results, ptrnALB, 'albedo')
            self.parseMultiParamFld(contents, i, results, ptrnBRDF, 'brdf', Nλ=Nλ) # GRASP cox-munk doens't require 2nd & 3rd parameters to be retrieved at all λ, not tested for BRDFs
            self.parseMultiParamFld(contents, i, results, ptrnBPDF, 'bpdf')
            self.parseMultiParamFld(contents, i, results, ptrnWater, 'wtrSurf', Nλ=Nλ)
            i += 1
        return results

    def parsePhaseMatrix(self, contents, wavelengths): # wavelengths is need here specificly b/c PM elements don't give index (only value in um)
        results = self.parseOutDateTime(contents)
        ptrnPMall = re.compile('^[ ]*Phase Matrix[ ]*$')
        ptrnPMfit = re.compile('^[ ]*ipix=([0-9]+)[ ]+yymmdd = [0-9]+-[0-9]+-[0-9]+[ ]+hhmmss[ ]*=[ ]*[0-9][0-9]:[0-9][0-9]:[0-9][0-9][ ]*$')
        ptrnPMfitWave = re.compile('^[ ]*wl=[ ]*([0-9]+.[0-9]+)[ ]+isd=([0-9]+)[ ]+sca=')
        FITfnd = False
        Nang = 181
        skipFlds = 1 # first field is just angle number
        pixInd = -1
        i = 0
        while i < len(contents):
            if not ptrnPMall.match(contents[i]) is None: # We found fitting data
                FITfnd = True
            pixMatch = ptrnPMfit.match(contents[i]) if FITfnd else None
            if pixMatch is not None: pixInd = int(pixMatch.group(1))-1 # We found a single pixel
            pixMatchWv = ptrnPMfitWave.match(contents[i]) if (FITfnd and pixInd>=0) else None
            if pixMatchWv is not None:  # We found a single pixel & wavelength & isd group
                l = np.nonzero(np.isclose(float(pixMatchWv.group(1)), wavelengths))[0][0]
                isd = int(pixMatchWv.group(2))-1# this is isdGRASP-1, ie. indexed at zero)
                flds = [s.replace('/','o') for s in contents[i+1].split()[skipFlds:]]
                PMdata = np.array([ln.split() for ln in contents[i+2:i+2+Nang]], np.float64)
                for j,fld in enumerate(flds):
                    if fld not in results[pixInd]: results[pixInd][fld] = np.zeros([Nang,0,0]) # OR JUST MAKE THIS A 3D ARRAY
                    if results[pixInd][fld].shape[1] == isd: # make bigger along isd dim
                        shp = results[pixInd][fld].shape
                        results[pixInd][fld] = np.concatenate((results[pixInd][fld], np.full([shp[0], 1, shp[2]], np.nan)), axis=1)
                    if results[pixInd][fld].shape[2] == l: # make one larger along lambda dim
                        shp = results[pixInd][fld].shape
                        results[pixInd][fld] = np.concatenate((results[pixInd][fld], np.full([shp[0], shp[1], 1], np.nan)), axis=2)
                    results[pixInd][fld][:,isd,l] = PMdata[:,j+skipFlds]
            i += 1
        return results

    def calcAsymParam(self, results): # Calculate the total asymmetry parameter, and lidar ratios while we're at it
        for rslt in results: # loop over pixels
            if np.all([fld in rslt for fld in ['p11', 'angle','aod','ssa','aodMode','ssaMode']]):
                rslt['g'] = np.empty(rslt['aod'].shape)
                useLidarRatioFromGRASP = 'LidarRatio' in rslt
                if not useLidarRatioFromGRASP: rslt['LidarRatio'] = np.empty(rslt['aod'].shape)
                if 'p22' in rslt: rslt['LidarDepol'] = np.empty(rslt['aod'].shape)
                rslt['gMode'] = np.empty(rslt['aodMode'].shape)
                rslt['LidarRatioMode'] = np.empty(rslt['aodMode'].shape)
                if 'p22' in rslt: rslt['LidarDepolMode'] = np.empty(rslt['aodMode'].shape)
                for l in range(rslt['p11'].shape[-1]): # loop over wavelength
                    for m,ssaMode in enumerate(rslt['ssaMode'][:,l]): # loop over mode
                        angRad = rslt['angle'][:,m,l]/180*np.pi
                        rslt['gMode'][m,l] = np.trapz(rslt['p11'][:,m,l]*np.cos(angRad)*np.sin(angRad), angRad)/2
                        rslt['LidarRatioMode'][m,l] =4*np.pi/(ssaMode*rslt['p11'][-1,m,l]) # we assume the last angle is θ=180°
                        rslt['LidarDepolMode'][m,l] = (rslt['p11'][-1,m,l] - rslt['p22'][-1,m,l])/(rslt['p11'][-1,m,l] + rslt['p22'][-1,m,l]) # we assume the last angle is θ=180°
                    scatWghts = rslt['ssaMode'][:,l]*rslt['aodMode'][:,l]
                    rslt['g'][l] = np.sum(rslt['gMode'][:,l]*scatWghts)/(rslt['ssa'][l]*rslt['aod'][l])
                    if 'p22' in rslt or not useLidarRatioFromGRASP:
                        F11bck = np.sum(scatWghts*rslt['p11'][-1,:,l])
                    if 'p22' in rslt:
                        F22bck = np.sum(scatWghts*rslt['p22'][-1,:,l])
                        rslt['LidarDepol'][l] = (F11bck-F22bck)/(F11bck+F22bck)
                    if not useLidarRatioFromGRASP:
                        rslt['LidarRatio'][l] = 4*np.pi*rslt['aod'][l]/F11bck # we assume the last angle is θ=180°

    def parseOutFit(self, contents, wavelengths):
        results = self.parseOutDateTime(contents)
        ptrnFIT = re.compile('^[ ]*[\*]+[ ]*FITTING[ ]*[\*]+[ ]*$')
        ptrnPIX = re.compile('^[ ]*pixel[ ]*#[ ]*([0-9]+)[ ]*wavelength[ ]*#[ ]*([0-9]+)[ ]*([0-9\.]+)[ ]*\(um\)')
        numericLn = re.compile('^[ ]*[0-9]+')
        ptrnHeader = re.compile('^[ ]*#[ ]*(sza[ ]*vis|Range_\[m\][ ]*meas_)')
        ptrnResid = re.compile('[ ]*noise[ ]*abs[ ]*rel[ ]*')
        i = 0
        skipFlds = 1 # the 1st field is just the measurement number
        FITfnd = False
        while i < len(contents):
            if not ptrnResid.match(contents[i]) is None: # next line is final value of the cost function
                pixInd = 0
                while numericLn.match(contents[i+1]):
                    results[pixInd]['costVal'] = float(re.search('^[ ]*[0-9\.]+', contents[i+1]).group())
                    i += 1
                    pixInd += 1
            if not ptrnFIT.match(contents[i]) is None: # We found fitting data
                FITfnd = True
            pixMatch = ptrnPIX.match(contents[i]) if FITfnd else None
            if pixMatch is not None:  # We found a single pixel & wavelength group
                pixInd = int(pixMatch.group(1))-1
        #        wvlInd = int(pixMatch.group(2))-1
                wvlVal = float(pixMatch.group(3))
                try:
                    wvlInd = np.nonzero(np.isclose(wavelengths, wvlVal, rtol=1e-3))[0][0]
                except IndexError:
                    msg = 'λ = %5.3f μm on line %d of GRASP output contents was not found in wavelengths!' % (wvlVal, i)
                    print('\x1b[1;31m'+msg+'\x1b[0m')
                while not ptrnHeader.match(contents[i+2]) is None: # loop over measurement types
                    flds = [s.replace('/','o').replace('_[m]','Lidar') for s in contents[i+2].split()[skipFlds:]]
                    lastLine = i+3
                    while lastLine < len(contents) and not numericLn.match(contents[lastLine]) is None:
                        lastLine += 1 # lastNumericInd+1
                    for ang,dataRow in enumerate(contents[i+3:lastLine]): # loop over angles
                        dArr = np.array(dataRow.split(), dtype='float64')[skipFlds:]
                        for j,fld in enumerate(flds):
                            if fld not in results[pixInd]: results[pixInd][fld] = np.array([]).reshape(0, len(wavelengths))
                            if results[pixInd][fld].shape[0] == ang: # need another angle row
                                nanRow = np.full((1, results[pixInd][fld].shape[1]), np.nan)
                                results[pixInd][fld] = np.block([[results[pixInd][fld]], [nanRow]])
                            results[pixInd][fld][ang, wvlInd] = dArr[j]
                    i = min(lastLine-2, len(contents)-3)
            i += 1
        return results

    def parseMultiParamFld(self, contents, i, results, ptrn, fdlName, fldName0=False, colOffset=0, Nλ=None):
        if i<len(contents) and not ptrn.match(contents[i]) is None: # RRI by aersol size mode
            singNumeric = re.compile('^[ ]*[0-9]+[ ]*$')
            numericLn = re.compile('^[ ]*[0-9]+')
            lastLine = i+1
            while not numericLn.match(contents[lastLine]) is None: lastLine += 1
            Nparams = 0
            for dataRow in contents[i+1:lastLine]:
                if not singNumeric.match(dataRow):
                    dArr = np.array(dataRow.split(), dtype='float64')
                    for k in range(len(results)): # this is looping over pixels
                        if fldName0:
                            results[k][fldName0] = np.append(results[k][fldName0], dArr[0+colOffset]) if fldName0 in results[k] else dArr[0+colOffset]
                        results[k][fdlName] = np.append(results[k][fdlName], dArr[k+1+colOffset]) if fdlName in results[k] else dArr[k+1+colOffset]
                else:
                    Nparams += 1
            if Nparams > 1:
                for k in range(len(results)): # seperate parameters from wavelengths
                    if Nλ is None or len(results[k][fdlName]) == Nλ*Nparams:
                        results[k][fdlName] = results[k][fdlName].reshape(Nparams,-1)
                    elif Nλ and len(results[k][fdlName]) == (Nλ+Nparams-1): # we take 1st parameter to contain spectrum, single value for 2nd, 3rd,...
                        top = results[k][fdlName][0:Nλ]
                        bot = [np.repeat(val, Nλ) for val in results[k][fdlName][Nλ:]]
                        results[k][fdlName] = np.vstack([top, bot])
                    elif k==0: # only show warning once
                        msg = 'Could not divide %s (len=%d) in Nparam=%d modes! Returning 1D array instead.'
                        warnings.warn(msg % (fdlName, len(results[k][fdlName]), Nparams))
            i = lastLine - 1

    def genSDATAHead(self, unqTimes):
        nx = max([pix.ix for pix in self.pixels])
        ny = max([pix.iy for pix in self.pixels])
        Nstr = ' %d %d %d : NX NY NT' % (nx, ny, len(unqTimes))
        return 'SDATA version 2.0\n%s\n' % Nstr

    def genCellHead(self, pixInd):
        nStr = '\n  %d   ' % len(pixInd)
        dtStr = self.pixels[pixInd[0]].dtObj.strftime('%Y-%m-%dT%H:%M:%SZ')
        endstr = ' %10.2f   0   0\n' % self.orbHght
        return nStr+dtStr+endstr


class pixel():
    def __init__(self, dtObj=None, ix=1, iy=1, lon=0, lat=0, masl=0, land_prct=100):
        """ dtObj - a datetime object corresponding to measurement time (also accepts matlab style datenum)
            masl - surface altitude in meters
            land_prct - % of land, in the range [0(sea)...100(land)] (matches format GRASP takes) """
        if type(dtObj) is np.float64: # we probably have a datenum
            dtObj = dt.fromordinal(int(dtObj)) + timedelta(days=dtObj % 1)
        self.dtObj = dtObj
        self.ix = ix
        self.iy = iy
        self.lon = lon
        self.lat = lat
        self.masl = masl
        if np.isclose(land_prct,1): warnings.warn('land_prct provided was 1.0 – this value is a percentage (100 -> completely land)')
        self.land_prct = land_prct
        self.nwl = 0
        self.measVals = []

    def addMeas(self, wl, msTyp=[], nbvm=[], sza=[], thtv=[], phi=[], msrmnts=[], errModel=None): # this is called once for each wavelength of data (see frmtMsg below)
        """Optimal input described by frmtMsg but method will expand thtv and phi if they have length len(msrmnts)/len(msTyp)"""
        assert wl not in [valDict['wl'] for valDict in self.measVals], 'Each measurement must have a unqiue wavelength!'
        newMeas = dict(wl=wl, nip=len(msTyp), meas_type=msTyp, nbvm=nbvm, sza=sza, thetav=thtv, phi=phi, measurements=msrmnts, errorModel=errModel)
        newMeas = self.formatMeas(newMeas)
        insertInd = np.nonzero([z['wl'] > newMeas['wl'] for z in self.measVals])[0] # we want to insert in order
        if len(insertInd)==0: # this is the longest wavelength so far, including the case w/ no measurements so far
            self.measVals.append(newMeas)
        else:
            self.measVals.insert(insertInd[0], newMeas)
        self.nwl += 1

    def populateFromRslt(self, rslt, radianceNoiseFun=None, dataStage='fit', verbose=False):
        """ This method will overwrite any previously existing data in the pixel at the following keys:
            meas_type, nbvm, nip, measurements, sza, thetav, phi, datetime, latitude, longitude, land_prct
            if self.meas == [] when called, populateFromRslt will add a measurement for each wvl in rslt
            radianceNoiseFun will override (and permanently set) self.measVals[n]['errorModel']
        """
        msTypMap = {'I':41, 'Q':42, 'U':43, 'LS':31, 'DP':35, 'VBS':39, 'VExt':36}
        msTyps = np.array([key.replace(dataStage+'_','') for key in rslt.keys() if dataStage in key]) # names of all keys with dataStage (e.g. "fit_")
        if 'QoI' in msTyps:
            rslt[dataStage+'_Q'] = rslt[dataStage+'_QoI']*rslt[dataStage+'_I']
            msTyps[msTyps=='QoI'] = 'Q'
        if 'UoI' in msTyps:
            rslt[dataStage+'_U'] = rslt[dataStage+'_UoI']*rslt[dataStage+'_I']
            msTyps[msTyps=='UoI'] = 'U'
        wvls = rslt['lambda']
        if self.nwl == 0: [self.addMeas(λ) for λ in wvls]
        for l, msDct in enumerate(self.measVals): # loop over wavelength
            msTypInd = np.nonzero([not np.isnan(rslt[dataStage+'_'+mt][:,l]).any() for mt in msTyps])[0] # inds of msTyps that are not NAN at current λ
            msDct['meas_type'] = [msTypMap[mt] for mt in msTyps[msTypInd]] # this is numberic key for measurements avaiable at current λ
            msTypsNowSorted = msTyps[msTypInd[np.argsort(msDct['meas_type'])]] # need msType names at this λ, sorted by above numeric measurement type keys
            msDct['nbvm'] = [len(rslt[dataStage+'_'+mt][:,l]) for mt in msTypsNowSorted] # number of measurement for each type (e.g. [10, 10, 10])
            msDct['meas_type'] = np.sort(msDct['meas_type'])
            msDct['nip'] = len(msDct['meas_type'])
            if np.all(msDct['meas_type'] < 40): # lidar data
                msDct['sza'] = 0.01 # we assume vertical lidar
                msDct['thetav'] = rslt['RangeLidar'][:,l]
                msDct['phi'] = np.repeat(0, len(msDct['thetav']))
            elif np.all(msDct['meas_type'] > 40): # polarimeter data
                msDct['sza'] = rslt['sza'][0,l] # GRASP/rslt dictionary return seperate SZA for every view, even though SDATA doesn't support it
                msDct['thetav'] = rslt['vis'][:,l]
                msDct['phi'] = rslt['fis'][:,l]
                if radianceNoiseFun: msDct['errorModel'] = radianceNoiseFun
            else:
                assert False, 'Both polarimeter and lidar data at the same wavelength is not supported.'
            if msDct['errorModel'] is not None:
                try:
                    msDct['measurements'] = msDct['errorModel'](l, rslt, verbose=verbose)
                except TypeError:
                    msDct['measurements'] = msDct['errorModel'](l, rslt)
            else:
                msDct['measurements'] = np.reshape([rslt[dataStage+'_'+msStr][:,l] for msStr in msTypsNowSorted], -1)
            msDct = self.formatMeas(msDct) # this will tile the above msTyp times
        if 'datetime' in rslt: self.dtObj = rslt['datetime']
        if 'latitude' in rslt: self.lat = rslt['latitude']
        if 'longitude' in rslt: self.lon = rslt['longitude']
        if 'land_prct' in rslt: self.land_prct = rslt['land_prct']
        if 'masl' in rslt: self.masl = rslt['masl']

    def formatMeas(self, newMeas, lowThresh=1e-10):
        frmtMsg = '\n\
            For more than one measurement type or viewing geometry pass msTyp, nbvm, thtv, phi and msrments as vectors: \n\
            len(msrments)=len(thtv)=len(phi)=sum(nbvm); len(msTyp)=len(nbvm) \n\
            msrments=[meas[msTyp[0],thtv[0],phi[0]], meas[msTyp[0],thtv[1],phi[1]],...,meas[msTyp[0],thtv[nbvm[0]],phi[nbvm[0]]],meas[msTyp[1],thtv[nbvm[0]+1],phi[nbvm[0]+1]],...]'
        assert newMeas['nip'] == len(newMeas['meas_type']), 'NIP should be equal to the number of measurement types'
        newMeas['meas_type'] = np.atleast_1d(newMeas['meas_type'])
        newMeas['nbvm'] = np.atleast_1d(newMeas['nbvm'])
        newMeas['thetav'] = np.atleast_1d(newMeas['thetav'])
        newMeas['phi'] = np.atleast_1d(newMeas['phi'])
        newMeas['measurements'] = np.atleast_1d(newMeas['measurements'])
        if len(newMeas['measurements']) > 0: # we have at least one measurement
            newMeas['measurements'][np.abs(newMeas['measurements']) < lowThresh] = lowThresh
            if len(newMeas['thetav']) == len(newMeas['measurements'])/newMeas['nip']: # viewing zenith not provided for each measurement type
                newMeas['thetav'] = np.tile(newMeas['thetav'], newMeas['nip'])
            if len(newMeas['phi']) == len(newMeas['measurements'])/newMeas['nip']: # relative azimuth not provided for each measurement type
                newMeas['phi'] = np.tile(newMeas['phi'], newMeas['nip'])
            newMeas['phi'] = newMeas['phi'] + 180*(np.array(newMeas['thetav'])<0) # GRASP doesn't like thetav < 0
            newMeas['thetav'] = np.abs(newMeas['thetav'])
        else:
            assert len(newMeas['phi'])==0 and len(newMeas['thetav'])==0, 'Angles were given but no measurement were present'
        if np.any(newMeas['phi'] < 0): warnings.warn('GRASP RT performance is hindered when phi < 0, values in the range 0 < phi < 360 are preferred.')
        assert newMeas['thetav'].shape[0]==newMeas['phi'].shape[0] and \
            newMeas['meas_type'].shape[0]==newMeas['nbvm'].shape[0] and \
            newMeas['nbvm'].sum()==newMeas['thetav'].shape[0], \
            'Each measurement must conform to the following format:' + frmtMsg
        return newMeas

    def genString(self):
        baseStrFrmt = '%2d %2d 1 0 0 %10.5f %10.5f %7.2f %6.2f %d' # everything up to meas fields
        baseStr = baseStrFrmt % (self.ix, self.iy, self.lon, self.lat, self.masl, self.land_prct, self.nwl)
        wlStr = " ".join(['%6.4f' % obj['wl'] for obj in self.measVals])
        nipStr = " ".join(['%d' % obj['nip'] for obj in self.measVals])
        allVals = np.block([obj['meas_type'] for obj in self.measVals])
        meas_typeStr = " ".join(['%d' % n for n in allVals])
        allVals = np.block([obj['nbvm'] for obj in self.measVals])
        nbvmStr = " ".join(['%d' % n for n in allVals])
        szaStr = " ".join(['%7.3f' % obj['sza'] for obj in self.measVals])
        allVals = np.block([obj['thetav'] for obj in self.measVals])
        thetavStr = " ".join(['%7.3f' % n for n in allVals])
        allVals = np.block([obj['phi'] for obj in self.measVals])
        phiStr = " ".join(['%7.3f' % n for n in allVals])
        allVals = np.block([obj['measurements'] for obj in self.measVals])
        measStr = " ".join(['%14.10f' % n for n in allVals])
        settingStr = '0 '*2*len(meas_typeStr.split(" "))
        measStrAll = " ".join((wlStr, nipStr, meas_typeStr, nbvmStr, szaStr, thetavStr, phiStr, measStr))
        return " ".join((baseStr, measStrAll, settingStr, '\n'))


class graspYAML():
    """Load, modify and/or store the contents of a YAML settings file."""

    def __init__(self, baseYAMLpath=None, workingYAMLpath=None, newTmpFile=False):
        """
        baseYAMLpath – the template YAML to take as a starting point
        workingYAMLpath – the path to write changed YAML values (baseYAMLpath is copied here)
            if None then the text in baseYAMLpath is overwritten by any changes
        newTmpFile –  workYAMLpath points to a file in the tmp directory; new file's name contains newTmpFile if it is a string
            NOTE: new YAML will have unique ID but is located in the common (shared) tmp directory
        """
        assert not (workingYAMLpath and not baseYAMLpath), 'baseYAMLpath must be provided to create a new YAML file at workingYAMLpath!'
        self.lambdaTypes = ['surface_water_CxMnk_iso_noPol',
                            'surface_water_cox_munk_iso',
                            'surface_land_brdf_ross_li',
                            'surface_land_polarized_maignan_breon',
                            'real_part_of_refractive_index_spectral_dependent',
                            'imaginary_part_of_refractive_index_spectral_dependent',
                            'lidar_calibration_coefficient']
        if workingYAMLpath:
            copyfile(baseYAMLpath, workingYAMLpath)
            self.YAMLpath = workingYAMLpath
        elif newTmpFile:
            randomID = hex(np.random.randint(0, 2**63-1))[2:] # needed to prevent identical FN w/ many parallel runs
            tag = newTmpFile if type(newTmpFile)==str else 'NEW'
            newFn = '%s_%s_%s.yml' % (os.path.basename(baseYAMLpath)[:-4], tag, randomID)
            self.YAMLpath = os.path.join(tempfile.gettempdir(), newFn)
            copyfile(baseYAMLpath, self.YAMLpath)
        else:
            self.YAMLpath = baseYAMLpath
        self.dl = None

    def setMultipleCharacteristics(self, vals, setField='value', Nlambda=None):
        fldNms = {
            'lgrnm':'size_distribution_lognormal',
            'sph':'sphere_fraction',
            'vol':'aerosol_concentration',
            'vrtHght':'vertical_profile_parameter_height',
            'vrtHghtStd':'vertical_profile_parameter_standard_deviation',
            'vrtProf':'vertical_profile_normalized',
            'n':'real_part_of_refractive_index_spectral_dependent',
            'k':'imaginary_part_of_refractive_index_spectral_dependent',
            'nCnst':'real_part_of_refractive_index_constant',
            'kCnst':'imaginary_part_of_refractive_index_constant',
            'brdf':'surface_land_brdf_ross_li',
            'bpdf':'surface_land_polarized_maignan_breon',
            'cxMnk':'surface_water_cox_munk_iso'}
        spectralFlds = ['n','k','brdf','bpdf','cxMnk']
        for key in vals.keys(): # loop over characteristics
            if key in spectralFlds:
                shapeValsKey = np.array(vals[key]).shape
                assert len(shapeValsKey) <= 2, '%s had %d dimensions – It should be 2D!' % (key, len(shapeValsKey))
                if Nlambda is None:
                    Nlambda = shapeValsKey[1]
                else:
                    assert Nlambda==shapeValsKey[1], '%s had a differnt Nλ than a previous characteristic!'
            for m in range(np.array(vals[key]).shape[0]): # loop over aerosol modes
                fldNm = '%s.%d.%s' % (fldNms[key], m+1, setField)
                self.access(fldNm, newVal=vals[key][m], write2disk=False, verbose=False) # verbose=False -> no wanrnings about creating a new mode
                if key=='vrtProf' and setField=='value': # adjust lambda will not fix this guy – NOTE: this overwrites min/max!
                    fldNm = '%s.%d.index_of_wavelength_involved' % (fldNms[key], m+1)
                    self.access(fldNm, newVal=np.zeros(len(vals[key][m]), dtype=int), write2disk=False)
                    fldNm = '%s.%d.min' % (fldNms[key], m+1)
                    self.access(fldNm, newVal=1e-9*np.ones(len(vals[key][m])), write2disk=False)
                    fldNm = '%s.%d.max' % (fldNms[key], m+1)
                    self.access(fldNm, newVal=2*np.ones(len(vals[key][m])), write2disk=False)
        if Nlambda: self.adjustLambda(Nlambda)
        self.writeYAML()

    def scrambleInitialGuess(self, fracOfSpace=1, skipTypes=['aerosol_concentration']):
        """Set a random initial guess for all types (excluding skipTypes), uniformly choosen from the range between min and max."""
        self.loadYAML()
        fracOfSpace = min(fracOfSpace, 0.999) # ensure we don't hit min/max exactly
        for char in self.dl['retrieval']['constraints'].values():
            if (not char['type'] in skipTypes) and char['retrieved']:
                for mode in np.array(list(char.values()))[['mode' in k for k in char.keys()]]: # only loop over keys containing "mode"
                    lowBnd = np.array(mode['initial_guess']['min'], dtype=float)
                    uprBnd = np.array(mode['initial_guess']['max'], dtype=float)
                    rngBnd = fracOfSpace*(uprBnd - lowBnd)/2
                    meanBnd = (lowBnd + uprBnd)/2
                    newGuess = np.random.uniform(meanBnd-rngBnd, meanBnd+rngBnd).tolist() # guess is spectrally flat relative to rng
                    mode['initial_guess']['value'] = newGuess
        self.writeYAML()

    def adjustLambda(self, Nlambda):
        """Change YAML settings to match a specific number of wavelenths, cutting and adding from the longest wavelength."""
        for lt in self.lambdaTypes: self._repeatElementsInField(fldName=lt, Nrepeats=Nlambda, λonly=True)  # loop over constraint types
        for n in range(len(self.access('retrieval.noises'))): # adjust the noise lambda as well
            m = 1
            while self.access('retrieval.noises.noise[%d].measurement_type[%d]' % (n+1, m)):
                fldNm = 'retrieval.noises.noise[%d].measurement_type[%d].index_of_wavelength_involved' % (n+1, m)
                orgVal = self.access(fldNm)
                if len(orgVal) >= Nlambda:
                    newVal = orgVal[0:Nlambda]
                else:
                    rpts = Nlambda - len(orgVal)
                    newVal = orgVal + np.r_[(orgVal[-1]+1):(orgVal[-1]+1+rpts)].tolist()
                self.access(fldNm, newVal, write2disk=False)
                m += 1
        self.writeYAML()

    def _repeatElementsInField(self, fldName, Nrepeats, λonly=False):
        """This will cut/repeat using the last element of characteristic fldName so that the number of entries is Nrepeats
            NOTE: This is a helper function that DOES NOT WRITE CHANGES TO THE FILE """
        assert np.issubdtype(type(Nrepeats), np.integer), 'Nrepeats must be an integer!'
        m = 1
        while self.access('%s.%d' % (fldName, m)): # loop over each mode
            λField = self.access('%s.%d.index_of_wavelength_involved' % (fldName, m))[0] > 0 # otherwise yaml specified [0] implying the parameter should be spectrally invarient
            if not λonly or λField:
                for f in ['index_of_wavelength_involved', 'value', 'min', 'max']:  # loop over each field
                    orgVal = self.access('%s.%d.%s' % (fldName, m, f))
                    if len(orgVal) >= Nrepeats:
                        self.access('%s.%d.%s' % (fldName, m, f), orgVal[0:Nrepeats], write2disk=False)
                    else:
                        rpts = Nrepeats - len(orgVal)
                        if f == 'index_of_wavelength_involved' and λField:
                            newVal = orgVal + np.r_[(orgVal[-1]+1):(orgVal[-1]+1+rpts)].tolist()
                        else:
                            newVal = orgVal + np.repeat(orgVal[-1], rpts).tolist()
                        self.access('%s.%d.%s' % (fldName, m, f), newVal, write2disk=False)
            m += 1

    def adjustVertBins(self, Nbins):
        self._repeatElementsInField(fldName='vertical_profile_normalized', Nrepeats=Nbins)
        self.writeYAML()

    def access(self, fldPath, newVal=None, write2disk=True, verbose=True): # will also return fldPath value if newVal=None
        if isinstance(newVal, np.ndarray):  # yaml module doesn't handle numby array gracefully
            newVal = newVal.tolist()
        elif isinstance(newVal, list): # check for regular list with numpy values
            for i, val in enumerate(newVal):
                if type(val).__module__ == np.__name__: newVal[i] = val.item()
        elif type(newVal).__module__ == np.__name__: # just a single value of a numpy type
            newVal = newVal.item()
        self.loadYAML()
        fldPath = self.exapndFldPath(fldPath)
        prsntVal = self.YAMLrecursion(self.dl, np.array(fldPath.split('.')), newVal)
        if not prsntVal and newVal: # we were supposed to change a value but the field wasn't there
            if verbose: warnings.warn('%s not found at specified location in YAML' % fldPath)
            mtch = re.match('retrieval.constraints.characteristic\[[0-9]+\].mode\[([0-9]+)\]', fldPath)
            if mtch: # we may still be able to add the value if we append a mode
                lastModePath = np.r_[fldPath.split('.')[0:3] + ['mode[%d]' % (int(mtch.group(1))-1)] + fldPath.split('.')[4:]]
                if self.YAMLrecursion(self.dl, lastModePath): # this field does exist in the previous mode, we will copy it
                    if verbose: print('The field does exists in previous mode of the same characterisitics, using it as a template to append a new mode...')
                    lastModeVal = self.YAMLrecursion(self.dl, lastModePath[0:4])
                    self.dl['retrieval']['constraints'][fldPath.split('.')[2]]['mode[%d]' % int(mtch.group(1))] = copy.deepcopy(lastModeVal)
                    prsntVal = self.YAMLrecursion(self.dl, np.array(fldPath.split('.')), newVal) # new mode exist now, write value to it
                    if not 'mode[%d]' % int(mtch.group(1)) in self.dl['retrieval']['phase_matrix']['radius'].keys(): # phase_matrix radius not present from this mode
                        lstModeRadius = self.dl['retrieval']['phase_matrix']['radius']['mode[%d]' % (int(mtch.group(1))-1)] # we copy it from previous mode
                        self.dl['retrieval']['phase_matrix']['radius']['mode[%d]' % int(mtch.group(1))] = copy.deepcopy(lstModeRadius)
        if newVal and write2disk: self.writeYAML() # if no change was made no need to re-write the file
        return prsntVal

    def exapndFldPath(self, fldPath):
        self.loadYAML()
        if fldPath == 'path_to_internal_files': # <-SHORTCUT: fldPath='path_to_internal_files'
            return 'retrieval.general.path_to_internal_files'
        if fldPath == 'stop_before_performing_retrieval': # <-SHORTCUT:
            return 'retrieval.convergence.stop_before_performing_retrieval'
        if fldPath == 'stream_fn': # <-SHORTCUT:
            return 'output.segment.stream'
        if fldPath == 'sdata_fn': # <-SHORTCUT:
            return 'input.file'
        charN = np.nonzero([val['type'] in fldPath for val in self.dl['retrieval']['constraints'].values()])[0]
        if charN.size > 0: # <-SHORTCUT: any type, ex. fldPath='aerosol_concentration' (set mode 1, value)
            fPvct = fldPath.split('.') # OR fldPath='aerosol_concentration.2' (mode 2, value)
            mode = fPvct[1] if len(fPvct) > 1 else 1 # OR fldPath='aerosol_concentration.2.min' (mode 3, min)
            fld = fPvct[2] if len(fPvct) > 2 else 'value'
            return 'retrieval.constraints.characteristic[%d].mode[%s].initial_guess.%s' % (charN[0]+1, mode, fld)
        return fldPath

    def writeYAML(self):
        with open(self.YAMLpath, 'w') as outfile:
            yaml.dump(self.dl, outfile, default_flow_style=None, indent=4, width=1000, sort_keys=False)

    def loadYAML(self):
        assert self.YAMLpath, 'You must provide a YAML file path to perform a task utilizing a YAML file!'
        if not self.dl:
            assert os.path.isfile(self.YAMLpath), 'The file '+self.YAMLpath+' does not exist!'
            with open(self.YAMLpath, 'r') as stream:
                try:
                    self.dl = yaml.load(stream, Loader=yaml.FullLoader)
                except AttributeError:
                    warnings.warn('Could not call yaml with FullLoader option. Is your conda up to date?')
                    self.dl = yaml.load(stream)

    def YAMLrecursion(self, yamlDict, fldPath, newVal=None):
        if not fldPath[0] in yamlDict: return None
        if fldPath.shape[0] > 1:
            return self.YAMLrecursion(yamlDict[fldPath[0]], fldPath[1::], newVal)
        if newVal is not None: yamlDict[fldPath[0]] = newVal
        return yamlDict[fldPath[0]]
