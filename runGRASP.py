#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import os.path
import warnings
import yaml # may require `conda install pyyaml`
import re
import time
import pickle
import copy
import numpy as np
import pandas as pd
import miscFunctions as mf
from datetime import datetime as dt # we want datetime.datetime
from datetime import timedelta
from shutil import copyfile
from subprocess import Popen,PIPE
from scipy.stats import gaussian_kde
try:
    import matplotlib.pyplot as plt
    pltLoad = True
except ImportError:
    pltLoad = False

class graspDB(object):
    def __init__(self, graspRunObjs=[], maxCPU=None, maxT=None):
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
                gObj = graspRun(graspRunObjs.yamlObj.YAMLpath, graspRunObjs.orbHght/1e3, graspRunObjs.dirGRASP)
                endInd = min(strtInd+grspChnkSz, Npix)
                for ind in range(strtInd, endInd):
                    gObj.addPix(graspRunObjs.pixels[ind])
                self.grObjs.append(gObj)
        else:
            assert not graspRunObjs, 'graspRunObj must be either a list or graspRun object!'
        
    def processData(self, maxCPUs=None, binPathGRASP=None, savePath=False, krnlPathGRASP=None, nodesSLURM=0):
        if not maxCPUs:
            maxCPUs = self.maxCPU if self.maxCPU else 2
        usedDirs = []
        t0 = time.time()
        for grObj in self.grObjs:
            if grObj.dirGRASP: # If not it will be deduced later when writing SDATA
                assert (not grObj.dirGRASP in usedDirs), "Each graspRun instance must use a unique directory!"
                usedDirs.append(grObj.dirGRASP)
            grObj.writeSDATA()
        if nodesSLURM==0:
            i = 0
            Nobjs = len(self.grObjs)
            pObjs = []
            while i < Nobjs:
                if sum([pObj.poll() is None for pObj in pObjs]) < maxCPUs:
                    print('Starting a new thread for graspRun index %d/%d' % (i+1,Nobjs))
                    pObjs.append(self.grObjs[i].runGRASP(True, binPathGRASP, krnlPathGRASP))
                    i+=1
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
            assert False, 'SLURM IS NOT YET SUPPORTED'
        self.rslts = []
#        [self.rslts.extend(grObj.readOutput()) for grObj in self.grObjs[~failedRuns]]
        [self.rslts.extend(self.grObjs[i].readOutput()) for i in np.nonzero(~failedRuns)[0]]
        dt = time.time() - t0
        print('%d pixels processed in %8.2f seconds (%5.2f pixels/second)' % (len(self.rslts), dt, len(self.rslts)/dt)) 
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
                 pltLabel=False, clnLayout=True): #clnLayout==False produces some speed up
        assert pltLoad, 'matplotlib could not be loaded, plotting features unavailable.'
        VarVal = self.getVarValues(VarNm, Ind, rsltInds)
        VarVal = VarVal[~pd.isnull(VarVal)] 
        assert VarVal.shape[0]>0, 'Zero valid matchups were found!'
        if customAx: plt.sca(customAx)
        plt.hist(VarVal, bins='auto')
        plt.xlabel(self.getLabelStr(VarNm, Ind))
        plt.ylabel('frequency')
        self.plotCleanUp(pltLabel, clnLayout)
        
    def scatterPlot(self, xVarNm, yVarNm, xInd=0, yInd=0, cVarNm=False, cInd=0, customAx=False,
                    logScl=False, Rstats=False, one2oneScale=False, FS=14, rsltInds=slice(None),
                    pltLabel=False, clnLayout=True): #clnLayout==False produces some speed up
        assert pltLoad, 'matplotlib could not be loaded, plotting features unavailable.'
        xVarVal = self.getVarValues(xVarNm, xInd, rsltInds)
        yVarVal = self.getVarValues(yVarNm, yInd, rsltInds)
        zeroErrStr = 'Values must be greater than zero for log scale!'
        noValPntsErrstr = 'Zero valid matchups were found!'
        if not cVarNm: #color by density
            vldInd = ~np.any((pd.isnull(xVarVal),pd.isnull(yVarVal)), axis=0)
            assert np.any(vldInd), noValPntsErrstr
            xVarVal = xVarVal[vldInd] 
            yVarVal = yVarVal[vldInd] 
            assert (not logScl) or ((xVarVal>0).all() and (yVarVal>0).all()), zeroErrStr
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
            assert (not logScl) or ((xVarVal>0).all() and (yVarVal>0).all()), zeroErrStr
        # GENERATE PLOTS
        if customAx: plt.sca(customAx)
        MS = 2 if xVarVal.shape[0]<5000 else 1
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
            textstr = 'N=%d\nR=%.3f\nRMS=%.3f\nbias=%.3f'%(len(xVarVal), Rcoef, RMSE, bias)
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
                 pltLabel=False, clnLayout=True): #clnLayout==False produces moderate speed up
        assert pltLoad, 'matplotlib could not be loaded, plotting features unavailable.'
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
            binEdge = np.histogram(xVarVal,bins='sturges')[1]
            binMid = (binEdge[1:]+binEdge[:-1])/2
        if 'aod' in xVarNm:
            binEdge = np.delete(binEdge,np.nonzero(binMid<0.005)[0]+1)
            binMid = binMid[binMid>=0.005]
            binEdge = np.delete(binEdge,np.nonzero(binMid>2.5)[0])
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
            txtStr = 'N=%d\nwithin 1xEE: %4.1f%%\nwithin 2xEE: %4.1f%%' % (Nval,inEE,in2EE)
            b = lambdaFuncEE(0)
            m = lambdaFuncEE(1) - b
            if np.all(y==m*x+b): # safe to assume EE function is linear
                txtStr = txtStr + '\nEE=%.2g+%.2gτ' % (b,m)
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
                warnings.warn('Ignoring index value %d for scalar %s' % (fldIndRaw,VarNm))
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
            if wvl>0: adTxt = adTxt + '%5.2g μm, ' % wvl
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
        assert self.rslts[0][VarNm].shape[0]>Ind[0], '1st index %d is out of bounds for variable %s' % (Ind[0],VarNm)
        if len(Ind)==2:
            assert self.rslts[0][VarNm].shape[1]>Ind[1], '2nd index %d is out of bounds for variable %s' % (Ind[1],VarNm)
        return np.array(Ind)

    def plotCleanUp(self, pltLabel=False, clnLayout=True):
        assert pltLoad, 'matplotlib could not be loaded, plotting features unavailable.'
        if pltLabel:
            plt.suptitle(pltLabel)
            if clnLayout: plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        elif clnLayout: 
            plt.tight_layout()

# can add self.AUX_dict[Npixel] dictionary list to instance w/ additional fields to port into rslts
class graspRun(object):
    def __init__(self, pathYAML=None, orbHghtKM=700, dirGRASP=False):
        self.pixels = [];
        self.pathSDATA = False;
        if pathYAML:
            self.dirGRASP = tempfile.mkdtemp() if not dirGRASP else dirGRASP
            print('Working in %s' % self.dirGRASP)
            newPathYAML = os.path.join(self.dirGRASP, os.path.basename(pathYAML))
            self.yamlObj = graspYAML(pathYAML, newPathYAML)
        else:
            assert not dirGRASP, 'You can not have a workin directory (dirGRASP) without a YAML file (pathYAML)!'
            self.dirGRASP = None
            self.yamlObj = graspYAML()
        self.orbHght = orbHghtKM*1000
        self.pObj = False
    
    def addPix(self, newPixel): # this is called once for each pixel
        self.pixels.append(copy.deepcopy(newPixel)) # deepcopy need to prevent changing via original pixel object outside of graspRun object
        
    def writeSDATA(self):
        if len(self.pixels) == 0:
            warnings.warn('You must call addPix() at least once before writting SDATA!')
            return False
        assert self.yamlObj.YAMLpath, 'You must initialize graspRun with a YAML file to write SDATA'
        self.pathSDATA = os.path.join(self.dirGRASP, self.yamlObj.access('sdata_fn'));
        assert (self.pathSDATA), 'Failed to read SDATA filename from '+self.yamlObj.YAMLpath
        unqTimes = np.unique([pix.dtNm for pix in self.pixels])
        SDATAstr = self.genSDATAHead(unqTimes)
        for unqTime in unqTimes:
            pixInd = np.nonzero(unqTime == [pix.dtNm for pix in self.pixels])[0]
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
        if krnlPathGRASP: self.access('path_to_internal_files', krnlPathGRASP)
        self.pObj = Popen([binPathGRASP, self.yamlObj.YAMLpath], stdout=PIPE)
        if not parallel:
            print('Running GRASP...')
            self.pObj.wait()
            self.pObj.stdout.close()
            self.invRslt = self.readOutput()          
        return self.pObj # returns Popen object, (PopenObj.poll() is not None) == True when complete
            
    def readOutput(self, customOUT=False): # customOUT is full path of unrelated SDATA file to read
        if not customOUT and not self.pObj:
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
        except:
            warnings.warn('Could not open %s\n   Returning empty list in place of output data...' % outputFN)
            return []
        rsltAeroDict, wavelengths = self.parseOutAerosol(contents)
        if not rsltAeroDict:
            warnings.warn('Aerosol data could not be read from %s\n  Returning empty list in place of output data...' % outputFN)
            return []
        rsltSurfDict = self.parseOutSurface(contents)
        rsltFitDict = self.parseOutFit(contents)
        rsltPMDict = self.parsePhaseMatrix(contents, wavelengths)
        rsltDict = []
        try:
            for aero, surf, fit, pm, aux in zip(rsltAeroDict, rsltSurfDict, rsltFitDict, rsltPMDict, self.AUX_dict):
                rsltDict.append({**aero, **surf, **fit, **pm, **aux})
        except AttributeError: # the user did not create AUX_dict
            for aero, surf, fit, pm in zip(rsltAeroDict, rsltSurfDict, rsltFitDict, rsltPMDict):
                rsltDict.append({**aero, **surf, **fit, **pm})
        return rsltDict
    
    def parseOutDateTime(self, contents):
        results = []
        ptrnDate = re.compile('^[ ]*Date[ ]*:[ ]+')
        ptrnTime = re.compile('^[ ]*Time[ ]*:[ ]+')
        ptrnLon = re.compile('^[ ]*Longitude[ ]*:[ ]+')
        ptrnLat = re.compile('^[ ]*Latitude[ ]*:[ ]+')
        i = 0
        LatLonLinesFnd = 0 # GRASP prints these twice, assume lat/lon are last lines
        while i<len(contents) and LatLonLinesFnd<2:
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
            i+=1
        if not len(results[-1].keys())==3:
            warnings.warn('Failure reading date/lat/lon from GRASP output!')
            return []
        return results

    def parseOutAerosol(self, contents):
        results = self.parseOutDateTime(contents)
        if len(results)==0:
            return []
        ptrnPSD = re.compile('^[ ]*(Radius \(um\),)?[ ]*Size Distribution dV\/dlnr \(normalized')
        ptrnLN = re.compile('^[ ]*Parameters of lognormal SD')
        ptrnVol = re.compile('^[ ]*Aerosol volume concentration')
        ptrnSPH = re.compile('^[ ]*% of spherical particles')
        ptrnHGNT = re.compile('^[ ]*Aerosol profile mean height')
        ptrnAOD = re.compile('^[ ]*Wavelength \(um\),[ ]+(Total_AOD|AOD_Total)')
        ptrnAODmode = re.compile('^[ ]*Wavelength \(um\),[ ]+AOD_Particle_mode')
        ptrnSSA = re.compile('^[ ]*Wavelength \(um\),[ ]+(SSA_Total|Total_SSA)')
        ptrnSSAmode = re.compile('^[ ]*Wavelength \(um\),[ ]+SSA_Particle_mode')
        ptrnRRI = re.compile('^[ ]*Wavelength \(um\), REAL Ref\. Index')
        ptrnIRI = re.compile('^[ ]*Wavelength \(um\), IMAG Ref\. Index')
        ptrnReff = re.compile('^[ ]*reff total[ ]*([0-9Ee.+\- ]+)[ ]*$') # this seems to have been removed in GRASP V0.8.2, atleast with >1 mode
        i = 0
        nsd = 0
        while i<len(contents):
            if not ptrnLN.match(contents[i]) is None: # lognormal PSD, these fields have unique form
                mtch = re.search('[ ]*rv \(um\):[ ]*', contents[i+1])
                rvArr = np.array(contents[i+1][mtch.end():-1].split(), dtype='float64')
                mtch = re.search('[ ]*ln\(sigma\):[ ]*', contents[i+2])
                sigArr = np.array(contents[i+2][mtch.end():-1].split(), dtype='float64')
                for k in range(len(results)):
                    results[k]['rv'] = np.append(results[k]['rv'], rvArr[k]) if 'rv' in results[k] else rvArr[k]
                    results[k]['sigma'] = np.append(results[k]['sigma'], sigArr[k]) if 'sigma' in results[k] else sigArr[k]
                i+=2
            if not ptrnReff.match(contents[i]) is None: # Reff, field has unique form
                Reffs = np.array(ptrnReff.match(contents[i]).group(1).split(), dtype='float64')
                for k,Reff in enumerate(Reffs):
                    results[k]['rEff'] = Reff     
            self.parseMultiParamFld(contents, i, results, ptrnAOD, 'aod', 'lambda')
            self.parseMultiParamFld(contents, i, results, ptrnPSD, 'dVdlnr', 'r')
            self.parseMultiParamFld(contents, i, results, ptrnVol, 'vol')
            self.parseMultiParamFld(contents, i, results, ptrnSPH, 'sph')
            self.parseMultiParamFld(contents, i, results, ptrnHGNT, 'height')   
            self.parseMultiParamFld(contents, i, results, ptrnAODmode, 'aodMode')
            self.parseMultiParamFld(contents, i, results, ptrnSSA, 'ssa')
            self.parseMultiParamFld(contents, i, results, ptrnSSAmode, 'ssaMode')
            self.parseMultiParamFld(contents, i, results, ptrnRRI, 'n')
            self.parseMultiParamFld(contents, i, results, ptrnIRI, 'k')
            i+=1
        if not results or not 'lambda' in results[0]:
            warnings.warn('Limited or no aerosol data found, returning incomplete dictionary...')
            return results
        wavelengths = results[0]['lambda']
        if 'aodMode' in results[0]:
            Nwvlth = 1 if np.isscalar(results[0]['aod']) else results[0]['aod'].shape[0]
            nsd = int(results[0]['aodMode'].shape[0]/Nwvlth)
            for rs in results: # seperate aerosol modes 
                rs['r'] = rs['r'].reshape(nsd,-1)
                rs['dVdlnr'] = rs['dVdlnr'].reshape(nsd,-1)
                rs['aodMode'] = rs['aodMode'].reshape(nsd,-1)
                rs['ssaMode'] = rs['ssaMode'].reshape(nsd,-1)
        if ('r' in results[0]) and np.all(results[0]['r'][0]==results[0]['r']): # check if all r value are same at all lambda, may remove this condition later but makes logic much more complicated
            for rs in results:
                dvdlnr = (rs['dVdlnr']*np.atleast_2d(rs['vol']).T).sum(axis=0)
                rs['rEffCalc'] = (mf.effRadius(rs['r'][0], dvdlnr))
        return results, wavelengths
    
    def parseOutSurface(self, contents):
        results = self.parseOutDateTime(contents)
        ptrnALB = re.compile('^[ ]*Wavelength \(um\),[ ]+Surface ALBEDO')
        ptrnBRDF = re.compile('^[ ]*Wavelength \(um\),[ ]+BRDF parameters')
        ptrnBPDF = re.compile('^[ ]*Wavelength \(um\),[ ]+BPDF parameters')
        ptrnWater = re.compile('^[ ]*Wavelength \(um\),[ ]+Water surface parameters')
        i = 0
        while i<len(contents):
            self.parseMultiParamFld(contents, i, results, ptrnALB, 'albedo')
            self.parseMultiParamFld(contents, i, results, ptrnBRDF, 'brdf')
            self.parseMultiParamFld(contents, i, results, ptrnBPDF, 'bpdf')
            self.parseMultiParamFld(contents, i, results, ptrnWater, 'wtrSurf')            
            i+=1
        return results

    def parsePhaseMatrix(self, contents, wavelengths): # wavelengths is need here specificly b/c PM elements don't give index (only value in um)
        results = self.parseOutDateTime(contents)
        ptrnPMall = re.compile('^[ ]*Phase Matrix[ ]*$')
        ptrnPMfit = re.compile('^[ ]*ipix=([0-9]+)[ ]+yymmdd = [0-9]+-[0-9]+-[0-9]+[ ]+hhmmss[ ]*=[ ]*[0-9][0-9]:[0-9][0-9]:[0-9][0-9][ ]*$')
        ptrnPMfitWave = re.compile('^[ ]*wl=[ ]*([0-9]+.[0-9]+)[ ]+isd=([0-9]+)[ ]+sca=')
        FITfnd = False
        Nang=181
        skipFlds = 1 # first field is just angle number
        pixInd = -1
        i=0
        while i<len(contents):
            if not ptrnPMall.match(contents[i]) is None: # We found fitting data
                FITfnd = True
            pixMatch = ptrnPMfit.match(contents[i]) if FITfnd else None
            if not pixMatch is None: pixInd = int(pixMatch.group(1))-1 # We found a single pixel
            pixMatchWv = ptrnPMfitWave.match(contents[i]) if (FITfnd and pixInd>=0) else None
            if not pixMatchWv is None:  # We found a single pixel & wavelength & isd group
                l = np.nonzero(np.isclose(float(pixMatchWv.group(1)), wavelengths))[0][0]
                isd = int(pixMatchWv.group(2))-1# this is isdGRASP-1, ie. indexed at zero)
                flds = [s.replace('/','o') for s in contents[i+1].split()[skipFlds:]]
                PMdata = np.array([ln.split() for ln in contents[i+2:i+2+Nang]], np.float64)
                for j,fld in enumerate(flds): 
                    if fld not in results[pixInd] : results[pixInd][fld] = np.zeros([Nang,0,0]) # OR JUST MAKE THIS A 3D ARRAY
                    if results[pixInd][fld].shape[1] == isd: # make bigger along isd dim
                        shp = results[pixInd][fld].shape
                        results[pixInd][fld] = np.concatenate((results[pixInd][fld], np.full([shp[0],1,shp[2]], np.nan)), axis=1)
                    if results[pixInd][fld].shape[2] == l: # make one larger along lambda dim
                        shp = results[pixInd][fld].shape
                        results[pixInd][fld] = np.concatenate((results[pixInd][fld], np.full([shp[0],shp[1],1], np.nan)), axis=2)
                    results[pixInd][fld][:,isd,l] = PMdata[:,j+skipFlds]
            i+=1
        return results
                
    def parseOutFit(self, contents):
        results = self.parseOutDateTime(contents)
        ptrnFIT = re.compile('^[ ]*[\*]+[ ]*FITTING[ ]*[\*]+[ ]*$')
        ptrnPIX = re.compile('^[ ]*pixel[ ]*#[ ]*([0-9]+)[ ]*wavelength[ ]*#[ ]*([0-9]+)')
        numericLn = re.compile('^[ ]*[0-9]+')
        i = 0
        skipFlds = 4 # the 1st 4 fields aren't interesting
        FITfnd = False
        while i<len(contents):
            if not ptrnFIT.match(contents[i]) is None: # We found fitting data
                FITfnd = True
            pixMatch = ptrnPIX.match(contents[i]) if FITfnd else None
            if not pixMatch is None:  # We found a single pixel & wavelength group
                pixInd = int(pixMatch.group(1))-1
                wvlInd = int(pixMatch.group(2))-1
                while not re.search('^[ ]*#[ ]*sza[ ]*vis', contents[i+2]) is None: # loop over measurement types
                    flds = [s.replace('/','o') for s in contents[i+2].split()[skipFlds:]]
                    lastLine = i+3 
                    while (lastLine < len(contents)) and not (numericLn.match(contents[lastLine]) is None): 
                        lastLine+=1 # lastNumericInd+1
                    for ang,dataRow in enumerate(contents[i+3:lastLine]): # loop over angles
                        dArr = np.array(dataRow.split(), dtype='float64')[skipFlds:]
                        for j,fld in enumerate(flds):
                            if fld not in results[pixInd]: results[pixInd][fld] = np.array([]).reshape(0,1)
                            if results[pixInd][fld].shape[1] == wvlInd: # need another column
                                nanCol = np.full((results[pixInd][fld].shape[0],1),np.nan)
                                results[pixInd][fld] = np.block([results[pixInd][fld],nanCol])
                            if results[pixInd][fld].shape[0] == ang: # need another angle row
                                nanRow = np.full((1,results[pixInd][fld].shape[1]),np.nan)
                                results[pixInd][fld] = np.block([[results[pixInd][fld]],[nanRow]])
                            results[pixInd][fld][ang,wvlInd] = dArr[j]
                    i=min(lastLine-2, len(contents)-3)
            i+=1
        return results
    
    def parseMultiParamFld(self, contents, i, results, ptrn, fdlName, fldName0=False):
        if not ptrn.match(contents[i]) is None: # RRI by aersol size mode
            singNumeric = re.compile('^[ ]*[0-9]+[ ]*$')
            numericLn = re.compile('^[ ]*[0-9]+')
            lastLine = i+1
            while not numericLn.match(contents[lastLine]) is None: lastLine+=1
            Nparams = 0
            for dataRow in contents[i+1:lastLine]:
                if not singNumeric.match(dataRow):
                    dArr = np.array(dataRow.split(), dtype='float64')
                    for k in range(len(results)):
                        if fldName0:
                            results[k][fldName0] = np.append(results[k][fldName0], dArr[0]) if fldName0 in results[k] else dArr[0]
                        results[k][fdlName] = np.append(results[k][fdlName], dArr[k+1]) if fdlName in results[k] else dArr[k+1]
                else:
                    Nparams += 1
            if Nparams > 1:
                for k in range(len(results)): # seperate parameters from wavelengths
                    results[k][fdlName] = results[k][fdlName].reshape(Nparams,-1)
            i = lastLine - 1
            
    def genSDATAHead(self, unqTimes):
        nx = max([pix.ix for pix in self.pixels])
        ny = max([pix.iy for pix in self.pixels])
        Nstr = ' %d %d %d : NX NY NT' % (nx, ny, len(unqTimes))
        return 'SDATA version 2.0\n%s\n' % Nstr
        
    def genCellHead(self, pixInd):
        nStr = '\n  %d   ' % len(pixInd)
        dtObjDay = dt.fromordinal(np.int(np.floor(self.pixels[pixInd[0]].dtNm)))
        dtObjTime = timedelta(seconds=np.remainder(self.pixels[pixInd[0]].dtNm, 1)*86400)
        dtObj = dtObjDay + dtObjTime
        dtStr = dtObj.strftime('%Y-%m-%dT%H:%M:%SZ')
        endstr = ' %10.2f   0   0\n' % self.orbHght
        return nStr+dtStr+endstr
        
   
class pixel(object):
    def __init__(self, dtNm, ix, iy, lon, lat, masl, land_prct):
        self.dtNm = dtNm
        self.ix = ix
        self.iy = iy
        self.lon = lon
        self.lat = lat
        self.masl = masl
        self.land_prct = land_prct
        self.nwl = 0
        self.measVals = []
         
    def addMeas(self, wl, msTyp, nbvm, sza, thtv, phi, msrmnts): # this is called once for each wavelength of data (see frmtMsg below)
        frmtMsg = '\n\
            For more than one measurement type or viewing geometry pass msTyp, nbvm, thtv, phi and msrments as vectors: \n\
            len(msrments)=len(thtv)=len(phi)=sum(nbvm); len(msTyp)=len(nbvm) \n\
            msrments=[meas[msTyp[0],thtv[0],phi[0]], meas[msTyp[0],thtv[1],phi[1]],...,meas[msTyp[0],thtv[nbvm[0]],phi[nbvm[0]]],meas[msTyp[1],thtv[nbvm[0]+1],phi[nbvm[0]+1]],...]'
        msTyp = np.atleast_1d(msTyp)
        nbvm = np.atleast_1d(nbvm)
        thtv = np.atleast_1d(thtv)
        phi = np.atleast_1d(phi) + 180*(np.array(thtv)<0) # GRASP doesn't like thtv < 0
        if np.any(phi<0): warnings.warn('GRASP RT performance is hindered when phi<0, values in the range 0<phi<360 are preferred.')
        thtv = np.abs(thtv) 
        msrmnts = np.atleast_1d(msrmnts)
        assert thtv.shape[0]==phi.shape[0] and msTyp.shape[0]==nbvm.shape[0] and nbvm.sum()==thtv.shape[0], 'Each measurement must conform to the following format:' + frmtMsg
        assert wl not in [valDict['wl'] for valDict in self.measVals], 'Each measurement must have a unqiue wavelength!' + frmtMsg
        newMeas = dict(wl=wl, nip=len(msTyp), meas_type=msTyp, nbvm=nbvm, sza=sza, thetav=thtv, phi=phi, measurements=msrmnts)
        self.measVals.append(newMeas)
        self.nwl += 1
         
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
    
class graspYAML(object):
    def __init__(self, baseYAMLpath=None, workingYAMLpath=None):
        assert not (workingYAMLpath and not baseYAMLpath), 'baseYAMLpath must be provided to create a new YAML file at workingYAMLpath!'
        if workingYAMLpath: 
            copyfile(baseYAMLpath, workingYAMLpath)
            self.YAMLpath = workingYAMLpath
        else:
            self.YAMLpath = baseYAMLpath
        self.dl = None
        
    def adjustLambda(self, Nlambda): # HINT: assumes all index of wavelength involved cover every wavelength
        lambdaTypes = ['surface_water_CxMnk_iso_noPol', 
               'surface_land_brdf_ross_li', 
               'surface_land_polarized_maignan_breon', 
               'real_part_of_refractive_index_spectral_dependent',
               'imaginary_part_of_refractive_index_spectral_dependent']
        for lt in lambdaTypes: # loop over constraint types
            m = 1;
            while self.access('%s.%d' % (lt,m)): # loop over each mode
                for f in ['index_of_wavelength_involved', 'value', 'min', 'max']:  # loop over each field
                    orgVal = self.access('%s.%d.%s' % (lt,m,f))
                    if len(orgVal) >= Nlambda:
                        self.access('%s.%d.%s' % (lt,m,f), orgVal[0:Nlambda], write2disk=False)
                    else:
                        rpts = Nlambda - len(orgVal)
                        if f=='index_of_wavelength_involved':
                            newVal = orgVal + np.r_[(orgVal[-1]+1):(orgVal[-1]+1+rpts)].tolist()
                        else:
                            newVal = orgVal + np.repeat(orgVal[-1],rpts).tolist()
                        self.access('%s.%d.%s' % (lt,m,f), newVal, write2disk=False)
                m+=1
        for n in range(len(self.access('retrieval.noises'))): # adjust the noise lambda as well
            m = 1
            while self.access('retrieval.noises.noise[%d].measurement_type[%d]' % (n+1,m)):
                fldNm = 'retrieval.noises.noise[%d].measurement_type[%d].index_of_wavelength_involved' % (n+1,m)
                orgVal = self.access(fldNm)
                if len(orgVal) >= Nlambda:
                    newVal = orgVal[0:Nlambda]
                else:
                    rpts = Nlambda - len(orgVal)                
                    newVal = orgVal + np.r_[(orgVal[-1]+1):(orgVal[-1]+1+rpts)].tolist()
                print(fldNm)
                print(newVal)
                self.access(fldNm, newVal, write2disk=False)
                m+=1
        self.writeYAML()
    
    def access(self, fldPath, newVal=None, write2disk=True, verbose=True): # will also return fldPath if newVal=None
        if isinstance(newVal,np.ndarray): newVal = newVal.tolist() # yaml module doesn't handle numby array gracefully 
        self.loadYAML()
        fldPath = self.exapndFldPath(fldPath)
        prsntVal = self.YAMLrecursion(self.dl, np.array(fldPath.split('.')), newVal)
        if not prsntVal and verbose and newVal: # we were supposed to change a value but the field wasn't there
            if verbose: warnings.warn('%s not found at specified location in YAML' % fldPath)
            mtch = re.match('retrieval.constraints.characteristic\[[0-9]+\].mode\[([0-9]+)\]', fldPath)
            if mtch: # we may still be able to add the value if we append a mode
                lastModePath = np.r_[fldPath.split('.')[0:3] + ['mode[%d]' % (int(mtch.group(1))-1)] + fldPath.split('.')[4:]]
                if self.YAMLrecursion(self.dl, lastModePath): # this field does exist in the previous mode, we will copy it
                    if verbose: print('The field does exists in previous mode of the same characterisitics, using it as a template to append a new mode...')
                    lastModeVal = self.YAMLrecursion(self.dl, lastModePath[0:4])
                    self.dl['retrieval']['constraints'][fldPath.split('.')[2]]['mode[%d]' % int(mtch.group(1))] = copy.deepcopy(lastModeVal)
                    prsntVal = self.YAMLrecursion(self.dl, np.array(fldPath.split('.')), newVal) # new mode exist now, write value to it
        if newVal and write2disk: self.writeYAML() # if no change was made no need to re-write the file
        return prsntVal
            
    def exapndFldPath(self, fldPath):
        self.loadYAML()
        if fldPath == 'path_to_internal_files': # <-SHORTCUT: fldPath='path_to_internal_files'
            return 'retrieval.general.path_to_internal_files'
        elif fldPath == 'stop_before_performing_retrieval': # <-SHORTCUT:
            return 'retrieval.convergence.stop_before_performing_retrieval'
        elif fldPath == 'stream_fn': # <-SHORTCUT:
            return 'output.segment.stream'
        elif fldPath == 'sdata_fn': # <-SHORTCUT:
            return 'input.file'
        charN = np.nonzero([val['type'] in fldPath for val in self.dl['retrieval']['constraints'].values()])[0]
        if charN.size>0: # <-SHORTCUT: any type, ex. fldPath='aerosol_concentration' (set mode 1, value)
            fPvct = fldPath.split('.') #  OR fldPath='aerosol_concentration.2' (mode 2, value)
            mode = fPvct[1] if len(fPvct) > 1 else 1 # OR fldPath='aerosol_concentration.2.min' (mode 3, min)
            fld = fPvct[2] if len(fPvct) > 2 else 'value'
            return 'retrieval.constraints.characteristic[%d].mode[%s].initial_guess.%s' % (charN[0]+1, mode, fld)
        return fldPath
        
    def writeYAML(self):
        with open(self.YAMLpath, 'w') as outfile:
                yaml.dump(self.dl, outfile, default_flow_style=None, indent=4, width=1000)
                
    def loadYAML(self):
        assert self.YAMLpath, 'You must provide a YAML file path to perform a task utilizing a YAML file!'
        if not self.dl:
            assert os.path.isfile(self.YAMLpath), 'The file '+self.YAMLpath+' does not exist!'
            with open(self.YAMLpath, 'r') as stream:
                self.dl = yaml.load(stream)
        
    def YAMLrecursion(self, yamlDict, fldPath, newVal=None):
        if not fldPath[0] in yamlDict: return None
        if fldPath.shape[0] > 1:
            return self.YAMLrecursion(yamlDict[fldPath[0]], fldPath[1::], newVal)
        else:
            if not newVal is None: yamlDict[fldPath[0]] = newVal
            return yamlDict[fldPath[0]]
                 
                 
                 
                 
                 
                 
                 